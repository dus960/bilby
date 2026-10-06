import os
import copy

import numpy as np
from scipy.integrate import cumulative_trapezoid, trapezoid, quad
from scipy.interpolate import InterpolatedUnivariateSpline, RegularGridInterpolator
from scipy.special import hyp2f1
from scipy.stats import norm

from ..core.prior import (
    PriorDict, Uniform, Prior, DeltaFunction, Gaussian, Interped, Constraint,
    conditional_prior_factory, PowerLaw, ConditionalLogUniform,
    ConditionalPriorDict, ConditionalBasePrior, BaseJointPriorDist, JointPrior,
    JointPriorDistError,
)
from ..core.utils import infer_args_from_method, logger, random, WrappedInterp1d as interp1d
from ..compat.utils import xp_wrap
from .conversion import (
    convert_to_lal_binary_black_hole_parameters,
    convert_to_lal_binary_neutron_star_parameters, generate_mass_parameters,
    generate_tidal_parameters, fill_from_fixed_priors,
    generate_all_bbh_parameters,
    chirp_mass_and_mass_ratio_to_total_mass,
    total_mass_and_mass_ratio_to_component_masses,
    component_masses_to_chirp_mass,
    lambda_1_lambda_2_to_lambda_tilde,
    lambda_1_lambda_2_to_delta_lambda_tilde,
    lambda_tilde_delta_lambda_tilde_to_lambda_1_lambda_2)
from .cosmology import get_cosmology, z_at_value
from .source import PARAMETER_SETS
from .utils import calculate_time_to_merger


DEFAULT_PRIOR_DIR = os.path.join(os.path.dirname(__file__), 'prior_files')


class BilbyPriorConversionError(Exception):
    pass


def convert_to_flat_in_component_mass_prior(result, fraction=0.25):
    """ Converts samples with a defined prior in chirp-mass and mass-ratio to flat in component mass by resampling with
    the posterior with weights defined as ratio in new:old prior values times the jacobian which for
    F(mc, q) -> G(m1, m2) is defined as J := m1^2 / mc

    Parameters
    ==========
    result: bilby.core.result.Result
        The output result complete with priors and posteriors
    fraction: float [0, 1]
        The fraction of samples to draw (default=0.25). Note, if too high a
        fraction of samples are draw, the reweighting will not be applied in
        effect.

    """
    if getattr(result, "priors") is not None:
        for key in ['chirp_mass', 'mass_ratio']:
            if key not in result.priors.keys():
                BilbyPriorConversionError("{} Prior not found in result object".format(key))
            if isinstance(result.priors[key], Constraint):
                BilbyPriorConversionError("{} Prior should not be a Constraint".format(key))
        for key in ['mass_1', 'mass_2']:
            if not isinstance(result.priors[key], Constraint):
                BilbyPriorConversionError("{} Prior should be a Constraint Prior".format(key))
    else:
        BilbyPriorConversionError("No prior in the result: unable to convert")

    result = copy.copy(result)
    priors = result.priors
    old_priors = copy.copy(result.priors)
    posterior = result.posterior

    for key in ['chirp_mass', 'mass_ratio']:
        priors[key] = Constraint(priors[key].minimum, priors[key].maximum, key, latex_label=priors[key].latex_label)
    for key in ['mass_1', 'mass_2']:
        priors[key] = Uniform(priors[key].minimum, priors[key].maximum, key, latex_label=priors[key].latex_label,
                              unit=r"$M_{\odot}$")

    weights = np.array(result.get_weights_by_new_prior(old_priors, priors,
                                                       prior_names=['chirp_mass', 'mass_ratio', 'mass_1', 'mass_2']))
    jacobian = posterior["mass_1"] ** 2 / posterior["chirp_mass"]
    weights = jacobian * weights
    result.posterior = posterior.sample(frac=fraction, weights=weights)

    logger.info("Resampling posterior to flat-in-component mass")
    effective_sample_size = sum(weights)**2 / sum(weights**2)
    n_posterior = len(posterior)
    if fraction > effective_sample_size / n_posterior:
        logger.warning(
            "Sampling posterior of length {} with fraction {}, but "
            "effective_sample_size / len(posterior) = {}. This may produce "
            "biased results"
            .format(n_posterior, fraction, effective_sample_size / n_posterior)
        )
    result.posterior = posterior.sample(frac=fraction, weights=weights, replace=True)
    result.meta_data["reweighted_to_flat_in_component_mass"] = True
    return result


class Cosmological(Interped):
    """
    Base prior class for cosmologically aware distance parameters.
    """

    @property
    def _default_args_dict(self):
        from astropy import units
        return dict(
            redshift=dict(name='redshift', latex_label='$z$', unit=None),
            luminosity_distance=dict(
                name='luminosity_distance', latex_label='$d_L$', unit=units.Mpc),
            comoving_distance=dict(
                name='comoving_distance', latex_label='$d_C$', unit=units.Mpc))

    def __init__(self, minimum, maximum, cosmology=None, name=None,
                 latex_label=None, unit=None, boundary=None):
        """

        Parameters
        ----------
        minimum: float
            The minimum value for the distribution.
        maximum: float
            The maximum value for the distribution.
        cosmology: Union[None, str, dict, astropy.cosmology.FLRW]
            The cosmology to use, see `bilby.gw.cosmology.set_cosmology`
            for more details.
            If :code:`None`, the default cosmology will be used.
        name: str
            The name of the parameter, should be one of
            :code:`[luminosity_distance, comoving_distance, redshift]`.
        latex_label: str
            Latex string to use in plotting, if :code:`None`, this will
            be set automatically using the name.
        unit: Union[str, astropy.units.Unit]
            The unit of the maximum and minimum values, :code:`default=Mpc`.
        boundary: str
            The boundary condition to apply to the prior when sampling.
        """
        from astropy import units
        self.cosmology = get_cosmology(cosmology)
        if name not in self._default_args_dict:
            raise ValueError(
                "Name {} not recognised. Must be one of luminosity_distance, "
                "comoving_distance, redshift".format(name))
        self.name = name
        label_args = self._default_args_dict[self.name]
        if latex_label is not None:
            label_args['latex_label'] = latex_label
        if unit is not None:
            if not isinstance(unit, units.Unit):
                unit = units.Unit(unit)
            label_args['unit'] = unit
        self.unit = label_args['unit']
        self._minimum = dict()
        self._maximum = dict()
        self.minimum = minimum
        self.maximum = maximum
        if name == 'redshift':
            xx, yy = self._get_redshift_arrays()
        elif name == 'comoving_distance':
            xx, yy = self._get_comoving_distance_arrays()
        elif name == 'luminosity_distance':
            xx, yy = self._get_luminosity_distance_arrays()
        else:
            raise ValueError('Name {} not recognized.'.format(name))
        super(Cosmological, self).__init__(xx=xx, yy=yy, minimum=minimum, maximum=maximum,
                                           boundary=boundary, **label_args)

    @property
    def minimum(self):
        return self._minimum[self.name]

    @minimum.setter
    def minimum(self, minimum):
        if (self.name in self._minimum) and (minimum < self.minimum):
            self._set_limit(value=minimum, limit_dict=self._minimum, recalculate_array=True)
        else:
            self._set_limit(value=minimum, limit_dict=self._minimum)

    @property
    def maximum(self):
        return self._maximum[self.name]

    @maximum.setter
    def maximum(self, maximum):
        if (self.name in self._maximum) and (maximum > self.maximum):
            self._set_limit(value=maximum, limit_dict=self._maximum, recalculate_array=True)
        else:
            self._set_limit(value=maximum, limit_dict=self._maximum)

    def _set_limit(self, value, limit_dict, recalculate_array=False):
        """
        Set either of the limits for redshift, luminosity, and comoving distances

        Parameters
        ==========
        value: float
            Limit value in current class' parameter
        limit_dict: dict
            The limit dictionary to modify in place
        recalculate_array: boolean
            Determines if the distance arrays are recalculated
        """
        cosmology = get_cosmology(self.cosmology)
        limit_dict[self.name] = value
        if self.name == 'redshift':
            limit_dict['luminosity_distance'] = \
                cosmology.luminosity_distance(value).value
            limit_dict['comoving_distance'] = \
                cosmology.comoving_distance(value).value
        elif self.name == 'luminosity_distance':
            if value == 0:
                limit_dict['redshift'] = 0
            else:
                limit_dict['redshift'] = z_at_value(
                    cosmology.luminosity_distance, value * self.unit
                )
            limit_dict['comoving_distance'] = (
                cosmology.comoving_distance(limit_dict['redshift']).value
            )
        elif self.name == 'comoving_distance':
            if value == 0:
                limit_dict['redshift'] = 0
            else:
                limit_dict['redshift'] = z_at_value(
                    cosmology.comoving_distance, value * self.unit
                )
            limit_dict['luminosity_distance'] = (
                cosmology.luminosity_distance(limit_dict['redshift']).value
            )
        if recalculate_array:
            if self.name == 'redshift':
                self.xx, self.yy = self._get_redshift_arrays()
            elif self.name == 'comoving_distance':
                self.xx, self.yy = self._get_comoving_distance_arrays()
            elif self.name == 'luminosity_distance':
                self.xx, self.yy = self._get_luminosity_distance_arrays()
        try:
            self._update_instance()
        except (AttributeError, KeyError):
            pass

    def get_corresponding_prior(self, name=None, unit=None):
        """
        Utility method to convert between equivalent redshift, comoving
        distance, and luminosity distance priors.

        Parameters
        ----------
        name: str
            The name of the new prior, this should be one of
            :code:`[luminosity_distance, comoving_distance, redshift]`.
        unit: Union[str, astropy.units.Unit]
            The unit of the output prior.

        Returns
        -------
        The corresponding prior.
        """
        subclass_args = infer_args_from_method(self.__init__)
        args_dict = {key: getattr(self, key) for key in subclass_args}
        self._convert_to(new=name, args_dict=args_dict)
        if unit is not None:
            args_dict['unit'] = unit
        return self.__class__(**args_dict)

    def _convert_to(self, new, args_dict):
        args_dict.update(self._default_args_dict[new])
        args_dict['minimum'] = self._minimum[args_dict['name']]
        args_dict['maximum'] = self._maximum[args_dict['name']]

    def _get_comoving_distance_arrays(self):
        zs, p_dz = self._get_redshift_arrays()
        dc_of_z = self.cosmology.comoving_distance(zs).value
        ddc_dz = np.gradient(dc_of_z, zs)
        p_dc = p_dz / ddc_dz
        return dc_of_z, p_dc

    def _get_luminosity_distance_arrays(self):
        zs, p_dz = self._get_redshift_arrays()
        dl_of_z = self.cosmology.luminosity_distance(zs).value
        ddl_dz = np.gradient(dl_of_z, zs)
        p_dl = p_dz / ddl_dz
        return dl_of_z, p_dl

    def _get_redshift_arrays(self):
        raise NotImplementedError

    @classmethod
    def from_repr(cls, string):
        if "FlatLambdaCDM" in string:
            logger.warning(
                "Cosmological priors cannot be loaded from a string. "
                "If the prior has a name, use that instead."
            )
            return string
        else:
            return cls._from_repr(string)

    def get_instantiation_dict(self):
        from astropy import units
        from .cosmology import get_available_cosmologies
        available = get_available_cosmologies()
        instantiation_dict = super().get_instantiation_dict()
        if self.cosmology.name in available:
            instantiation_dict['cosmology'] = self.cosmology.name
        if isinstance(self.unit, units.Unit):
            instantiation_dict['unit'] = self.unit.to_string()
        return instantiation_dict


class UniformComovingVolume(Cosmological):
    r"""
    Prior that is uniform in the comoving volume.

    Note that this does not account for time dilation in the source frame
    use `bilby.gw.prior.UniformSourceFrame` to include that effect.

    .. math::

        p(z) \propto \frac{d V_{c}}{dz}
    """

    def _get_redshift_arrays(self):
        zs = np.linspace(self._minimum['redshift'] * 0.99,
                         self._maximum['redshift'] * 1.01, 1000)
        p_dz = self.cosmology.differential_comoving_volume(zs).value
        return zs, p_dz


class UniformSourceFrame(Cosmological):
    r"""
    Prior for redshift which is uniform in comoving volume and source frame
    time.

    .. math::

        p(z) \propto \frac{1}{1 + z}\frac{dV_{c}}{dz}

    The extra :math:`1+z` is due to doppler shifting of the source frame time.
    """

    def _get_redshift_arrays(self):
        zs = np.linspace(self._minimum['redshift'] * 0.99,
                         self._maximum['redshift'] * 1.01, 1000)
        p_dz = self.cosmology.differential_comoving_volume(zs).value / (1 + zs)
        return zs, p_dz


class UniformInComponentsChirpMass(PowerLaw):
    r"""
    Prior distribution for chirp mass which is uniform in component masses.

    This is useful when chirp mass and mass ratio are sampled while the
    prior is uniform in component masses.

    .. math::

        p({\cal M}) \propto {\cal M}

    Notes
    -----
    This prior is intended to be used in conjunction with the corresponding
    :code:`bilby.gw.prior.UniformInComponentsMassRatio`.
    """

    def __init__(self, minimum, maximum, name='chirp_mass',
                 latex_label=r'$\mathcal{M}$', unit=None, boundary=None):
        """
        Parameters
        ==========
        minimum : float
            The minimum of chirp mass
        maximum : float
            The maximum of chirp mass
        name: see superclass
        latex_label: see superclass
        unit: see superclass
        boundary: see superclass
        """
        super(UniformInComponentsChirpMass, self).__init__(
            alpha=1., minimum=minimum, maximum=maximum,
            name=name, latex_label=latex_label, unit=unit, boundary=boundary)


class UniformInComponentsMassRatio(Prior):
    r"""
    Prior distribution for chirp mass which is uniform in component masses.

    This is useful when chirp mass and mass ratio are sampled while the
    prior is uniform in component masses.

    .. math::

        p(q) \propto \frac{(1 + q)^{2/5}}{q^{6/5}}

    Notes
    -----
    This prior is intended to be used in conjunction with the corresponding
    :code:`bilby.gw.prior.UniformInComponentsChirpMass`.
    """

    def __init__(self, minimum, maximum, name='mass_ratio', latex_label='$q$',
                 unit=None, boundary=None, equal_mass=False):
        """
        Parameters
        ==========
        minimum : float
            The minimum of mass ratio
        maximum : float
            The maximum of mass ratio
        name: see superclass
        latex_label: see superclass
        unit: see superclass
        boundary: see superclass
        equal_mass: bool
            Whether the likelihood being considered is expected to peak at
            equal masses. If True, the mapping described in Appendix A of
            arXiv:2111.13619 is used in the :code:`rescale` method.
            default=False

        """
        self.equal_mass = equal_mass
        super(UniformInComponentsMassRatio, self).__init__(
            minimum=minimum, maximum=maximum, name=name,
            latex_label=latex_label, unit=unit, boundary=boundary)
        self.norm = self._integral(maximum) - self._integral(minimum)
        qs = np.linspace(minimum, maximum, 1000)
        self.icdf = interp1d(
            self.cdf(qs), qs, kind='cubic',
            bounds_error=False, fill_value=(minimum, maximum))

    @staticmethod
    def _integral(q):
        return -5. * q**(-1. / 5.) * hyp2f1(-2. / 5., -1. / 5., 4. / 5., -q)

    def cdf(self, val):
        return (self._integral(val) - self._integral(self.minimum)) / self.norm

    def rescale(self, val):
        if self.equal_mass:
            val = 2 * np.minimum(val, 1 - val)
        return self.icdf(val)

    def prob(self, val):
        in_prior = (val >= self.minimum) & (val <= self.maximum)
        with np.errstate(invalid="ignore"):
            prob = (1. + val)**(2. / 5.) / (val**(6. / 5.)) / self.norm * in_prior
        return prob

    def ln_prob(self, val):
        with np.errstate(divide="ignore"):
            return np.log(self.prob(val))


class AlignedSpin(Interped):
    r"""
    Prior distribution for the aligned (z) component of the spin.

    This takes prior distributions for the magnitude and cosine of the tilt
    and forms a compound prior using a numerical convolution integral.

    .. math::

        \pi(\chi) = \int_{0}^{1} da \int_{-1}^{1} d\cos\theta
        \pi(a) \pi(\cos\theta) \delta(\chi - a \cos\theta)

    This is useful when using aligned-spin only waveform approximants.

    This is an extension of e.g., (A7) of https://arxiv.org/abs/1805.10457.
    For the simple case of uniform in magnitude and orientation priors, we use an
    analytic form of the PDF.
    """

    def __init__(
        self,
        a_prior=Uniform(0, 1),
        z_prior=Uniform(-1, 1),
        name=None,
        latex_label=None,
        unit=None,
        boundary=None,
        minimum=np.nan,
        maximum=np.nan,
        num_interp=None,
    ):
        """
        Parameters
        ==========
        a_prior: Prior
            Prior distribution for spin magnitude
        z_prior: Prior
            Prior distribution for cosine spin tilt
        name: see superclass
        latex_label: see superclass
        unit: see superclass
        num_interp: int
            The number of points to use in the interpolation. Defaults to 100,000
            for simple aligned priors and 10,000 for general priors.
        """
        self.a_prior = a_prior
        self.z_prior = z_prior
        chi_min = min(a_prior.maximum * z_prior.minimum,
                      a_prior.minimum * z_prior.maximum)
        chi_max = a_prior.maximum * z_prior.maximum
        if self._is_simple_aligned_prior:
            self.num_interp = 100_000 if num_interp is None else num_interp
            xx = np.linspace(chi_min, chi_max, self.num_interp)
            yy = - np.log(np.abs(xx) / a_prior.maximum) / (2 * a_prior.maximum)
        else:

            def integrand(aa, chi):
                """
                The integrand for the aligned spin (chi) probability density
                after performing the integral over spin orientation using a
                delta function identity.
                """
                return a_prior.prob(aa) * z_prior.prob(chi / aa) / aa

            self.num_interp = 10_000 if num_interp is None else num_interp
            xx = np.linspace(chi_min, chi_max, self.num_interp)
            yy = [
                quad(integrand, a_prior.minimum, a_prior.maximum, chi)[0]
                for chi in xx
            ]
        super().__init__(
            xx=xx,
            yy=yy,
            name=name,
            latex_label=latex_label,
            unit=unit,
            boundary=boundary,
            minimum=minimum,
            maximum=maximum,
        )

    @property
    def _is_simple_aligned_prior(self):
        return (
            isinstance(self.a_prior, Uniform)
            and isinstance(self.z_prior, Uniform)
            and self.z_prior.minimum == -1
            and self.z_prior.maximum == 1
        )


class ConditionalChiUniformSpinMagnitude(ConditionalLogUniform):
    r"""
    This prior characterizes the conditional prior on the spin magnitude given
    the aligned component of the spin  such that the marginal prior is uniform
    if the distribution of spin orientations is isotropic.

    .. math::

        p(a) &= \frac{1}{a_{\max}} \\
        p(\chi) &= - \frac{\ln(|\chi|)}{2 a_{\max}}  \\
        p(a | \chi) &\propto \frac{1}{a}
    """

    def __init__(self, minimum, maximum, name, latex_label=None, unit=None, boundary=None):
        super(ConditionalChiUniformSpinMagnitude, self).__init__(
            minimum=minimum, maximum=maximum, name=name, latex_label=latex_label, unit=unit, boundary=boundary,
            condition_func=self._condition_function)
        self._required_variables = [name.replace("a", "chi")]
        self.__class__.__name__ = "ConditionalChiUniformSpinMagnitude"
        self.__class__.__qualname__ = "ConditionalChiUniformSpinMagnitude"

    def _condition_function(self, reference_params, **kwargs):
        return dict(minimum=np.abs(kwargs[self._required_variables[0]]), maximum=reference_params["maximum"])

    def __repr__(self):
        return Prior.__repr__(self)

    def get_instantiation_dict(self):
        instantiation_dict = Prior.get_instantiation_dict(self)
        for key, value in self.reference_params.items():
            if key in instantiation_dict:
                instantiation_dict[key] = value
        return instantiation_dict


class ConditionalChiInPlane(ConditionalBasePrior):
    r"""
    This prior characterizes the conditional prior on the in-plane spin magnitude
    given the aligned component of the spin  such that the marginal prior is uniform
    if the distribution of spin orientations is isotropic.

    .. math::

        p(a) &= \frac{1}{a_{\max}} \\
        p(\chi_\perp) &= \frac{2 N \chi_\perp}{\chi^2 + \chi_\perp^2} \\
        N^{-1} &= 2 \ln\left(\frac{a_\max}{|\chi|}\right)
    """

    def __init__(self, minimum, maximum, name, latex_label=None, unit=None, boundary=None):
        super(ConditionalChiInPlane, self).__init__(
            minimum=minimum, maximum=maximum,
            name=name, latex_label=latex_label,
            unit=unit, boundary=boundary,
            condition_func=self._condition_function
        )
        self._required_variables = [name[:5]]
        self._reference_maximum = maximum
        self.__class__.__name__ = "ConditionalChiInPlane"
        self.__class__.__qualname__ = "ConditionalChiInPlane"

    def prob(self, val, **required_variables):
        self.update_conditions(**required_variables)
        chi_aligned = abs(required_variables[self._required_variables[0]])
        return (
            (val >= self.minimum) * (val <= self.maximum)
            * val
            / (chi_aligned ** 2 + val ** 2)
            / np.log(self._reference_maximum / chi_aligned)
        )

    def ln_prob(self, val, **required_variables):
        with np.errstate(divide="ignore"):
            return np.log(self.prob(val, **required_variables))

    def cdf(self, val, **required_variables):
        r"""
        .. math::
            \text{CDF}(\chi_\per) = N ln(1 + (\chi_\perp / \chi) ** 2)

        Parameters
        ----------
        val: (float, array-like)
            The value at which to evaluate the CDF
        required_variables: dict
            A dictionary containing the aligned component of the spin

        Returns
        -------
        (float, array-like)
            The value of the CDF

        """
        self.update_conditions(**required_variables)
        chi_aligned = abs(required_variables[self._required_variables[0]])
        return np.maximum(np.minimum(
            (val >= self.minimum) * (val <= self.maximum)
            * np.log(1 + (val / chi_aligned) ** 2)
            / 2 / np.log(self._reference_maximum / chi_aligned)
            , 1
        ), 0)

    def rescale(self, val, **required_variables):
        r"""
        .. math::
            \text{PPF}(\chi_\perp) = ((a_\max / \chi) ** (2x) - 1) ** 0.5 * \chi

        Parameters
        ----------
        val: (float, array-like)
            The value to rescale
        required_variables: dict
            Dictionary containing the aligned spin component

        Returns
        -------
        (float, array-like)
            The in-plane component of the spin
        """
        self.update_conditions(**required_variables)
        chi_aligned = abs(required_variables[self._required_variables[0]])
        return chi_aligned * ((self._reference_maximum / chi_aligned) ** (2 * val) - 1) ** 0.5

    def _condition_function(self, reference_params, **kwargs):
        with np.errstate(invalid="ignore"):
            maximum = np.sqrt(
                self._reference_maximum ** 2 - kwargs[self._required_variables[0]] ** 2
            )
        return dict(minimum=0, maximum=maximum)

    def __repr__(self):
        return Prior.__repr__(self)

    def get_instantiation_dict(self):
        instantiation_dict = Prior.get_instantiation_dict(self)
        for key, value in self.reference_params.items():
            if key in instantiation_dict:
                instantiation_dict[key] = value
        return instantiation_dict


class EOSCheck(Constraint):
    def __init__(self, minimum=-np.inf, maximum=np.inf):
        """
        Constraint used for EoS sampling. Converts the result of various
        checks on the EoS and its parameters into a prior that can reject
        unphysical samples. Necessary for EoS sampling.
        """

        super().__init__(minimum=minimum, maximum=maximum, name=None, latex_label=None, unit=None)

    def prob(self, val):
        """
        Returns the result of the equation of state check in the conversion function.
        """
        return val

    def ln_prob(self, val):

        if val:
            result = 0.0
        elif not val:
            result = -np.inf

        return result


class CBCPriorDict(ConditionalPriorDict):
    @property
    def minimum_chirp_mass(self):
        mass_1 = None
        mass_2 = None
        if "chirp_mass" in self:
            return self["chirp_mass"].minimum
        elif "mass_1" in self:
            mass_1 = self['mass_1'].minimum
            if "mass_2" in self:
                mass_2 = self['mass_2'].minimum
            elif "mass_ratio" in self:
                mass_2 = mass_1 * self["mass_ratio"].minimum
        if mass_1 is not None and mass_2 is not None:
            s = generate_mass_parameters(dict(mass_1=mass_1, mass_2=mass_2))
            return s["chirp_mass"]
        else:
            logger.warning("Unable to determine minimum chirp mass")
            return None

    @property
    def maximum_chirp_mass(self):
        mass_1 = None
        mass_2 = None
        if "chirp_mass" in self:
            return self["chirp_mass"].maximum
        elif "mass_1" in self:
            mass_1 = self['mass_1'].maximum
            if "mass_2" in self:
                mass_2 = self['mass_2'].maximum
            elif "mass_ratio" in self:
                mass_2 = mass_1 * self["mass_ratio"].maximum
        if mass_1 is not None and mass_2 is not None:
            s = generate_mass_parameters(dict(mass_1=mass_1, mass_2=mass_2))
            return s["chirp_mass"]
        else:
            logger.warning("Unable to determine maximum chirp mass")
            return None

    @property
    def minimum_component_mass(self):
        """
        The minimum component mass allowed for the prior dictionary.

        This property requires either:
        * a prior for :code:`mass_2`
        * priors for :code:`chirp_mass` and :code:`mass_ratio`

        Returns
        -------
        mass_2: float
            The minimum allowed component mass.
        """
        if "mass_2" in self:
            return self["mass_2"].minimum
        if "chirp_mass" in self and "mass_ratio" in self:
            total_mass = chirp_mass_and_mass_ratio_to_total_mass(
                self["chirp_mass"].minimum, self["mass_ratio"].minimum)
            _, mass_2 = total_mass_and_mass_ratio_to_component_masses(
                self["mass_ratio"].minimum, total_mass)
            return mass_2
        else:
            logger.warning("Unable to determine minimum component mass")
            return None

    def is_nonempty_intersection(self, pset):
        """ Check if keys in self exist in the parameter set

        Parameters
        ----------
        pset: str, set
            Either a string referencing a parameter set in PARAMETER_SETS or
            a set of keys
        """
        if isinstance(pset, str) and pset in PARAMETER_SETS:
            check_set = PARAMETER_SETS[pset]
        elif isinstance(pset, set):
            check_set = pset
        else:
            raise ValueError(f"pset {pset} not understood")
        if len(check_set.intersection(self.non_fixed_keys)) > 0:
            return True
        else:
            return False

    @property
    def spin(self):
        """ Return true if priors include any spin parameters """
        return self.is_nonempty_intersection("spin")

    @property
    def precession(self):
        """ Return true if priors include any precession parameters """
        return self.is_nonempty_intersection("precession_only")

    @property
    def measured_spin(self):
        """ Return true if priors include any measured_spin parameters """
        return self.is_nonempty_intersection("measured_spin")

    @property
    def intrinsic(self):
        """ Return true if priors include any intrinsic parameters """
        return self.is_nonempty_intersection("intrinsic")

    @property
    def extrinsic(self):
        """ Return true if priors include any extrinsic parameters """
        return self.is_nonempty_intersection("extrinsic")

    @property
    def sky(self):
        """ Return true if priors include any extrinsic parameters """
        return self.is_nonempty_intersection("sky")

    @property
    def distance_inclination(self):
        """ Return true if priors include any extrinsic parameters """
        return self.is_nonempty_intersection("distance_inclination")

    @property
    def mass(self):
        """ Return true if priors include any mass parameters """
        return self.is_nonempty_intersection("mass")

    @property
    def phase(self):
        """ Return true if priors include phase parameters """
        return self.is_nonempty_intersection("phase")

    @property
    def _cosmological_priors(self):
        return [
            key for key, prior in self.items() if isinstance(prior, Cosmological)
        ]

    @property
    def is_cosmological(self):
        """Return True if any of the priors are cosmological."""
        if self._cosmological_priors:
            return True
        else:
            return False

    @property
    def cosmology(self):
        """The cosmology used in the priors."""
        if self.is_cosmological:
            return self[self._cosmological_priors[0]].cosmology
        else:
            return None

    def check_valid_cosmology(self, error=True, warning=False):
        """Check that all cosmological priors use the same cosmology.

        .. versionadded:: 2.5.0

        Parameters
        ==========
        error: bool
            Whether to raise a ValueError on failure.
        warning: bool
            Whether to log a warning on failure.

        Returns
        =======
        bool: whether the cosmological priors are valid.

        Raises
        ======
        ValueError: if error is True and the cosmological priors are invalid.
        """
        cosmological_priors = self._cosmological_priors
        if not cosmological_priors:
            return True

        from astropy.cosmology import cosmology_equal

        cosmologies = [self[key].cosmology for key in self._cosmological_priors]
        if all(cosmology_equal(cosmologies[0], c, allow_equivalent=True) for c in cosmologies[1:]):
            return True

        message = (
            "All cosmological priors must use the same cosmology. "
            f"Found: {cosmologies}"
        )
        if warning:
            logger.warning(message)
            return False
        if error:
            raise ValueError(message)

    def validate_prior(self, duration, minimum_frequency, N=1000, error=True, warning=False):
        """ Validate the prior is suitable for use

        Parameters
        ==========
        duration: float
            The data duration in seconds
        minimum_frequency: float
            The minimum frequency in Hz of the analysis
        N: int
            The number of samples to draw when checking
        error: bool
            Whether to raise a ValueError on failure.
        warning: bool
            Whether to log a warning on failure.

        Returns
        =======
        bool: whether the template will fit within the segment duration
        """
        samples = self.sample(N)
        samples = generate_all_bbh_parameters(samples)
        durations = np.array([
            calculate_time_to_merger(
                frequency=minimum_frequency,
                mass_1=mass_1,
                mass_2=mass_2,
            )
            for (mass_1, mass_2) in zip(samples["mass_1"], samples["mass_2"])
        ])
        longest_duration = max(durations)
        if longest_duration < duration:
            return True
        message = (
            "Prior contains regions of parameter space in which the signal"
            f" is longer than the data duration {duration}s."
            f" Maximum estimated duration = {longest_duration:.1f}s."
        )
        if warning:
            logger.warning(message)
            return False
        if error:
            raise ValueError(message)


class BBHPriorDict(CBCPriorDict):
    def __init__(self, dictionary=None, filename=None, aligned_spin=False,
                 conversion_function=None):
        """ Initialises a Prior set for Binary Black holes

        Parameters
        ==========
        dictionary: dict, optional
            See superclass
        filename: str, optional
            See superclass
        conversion_function: func
            Function to convert between sampled parameters and constraints.
            By default this generates many additional parameters, see
            BBHPriorDict.default_conversion_function
        """
        if dictionary is None and filename is None:
            if aligned_spin:
                fname = 'aligned_spins_bbh.prior'
                logger.info('Using aligned spin prior')
            else:
                fname = 'precessing_spins_bbh.prior'
            filename = os.path.join(DEFAULT_PRIOR_DIR, fname)
            logger.info('No prior given, using default BBH priors in {}.'.format(filename))
        elif filename is not None:
            if not os.path.isfile(filename):
                filename = os.path.join(DEFAULT_PRIOR_DIR, filename)
        super(BBHPriorDict, self).__init__(dictionary=dictionary, filename=filename,
                                           conversion_function=conversion_function)

    def default_conversion_function(self, sample):
        """
        Default parameter conversion function for BBH signals.

        This generates:
        - the parameters passed to source.lal_binary_black_hole
        - all mass parameters

        It does not generate:
        - component spins
        - source-frame parameters

        Parameters
        ==========
        sample: dict
            Dictionary to convert

        Returns
        =======
        sample: dict
            Same as input
        """
        out_sample = fill_from_fixed_priors(sample, self)
        out_sample, _ = convert_to_lal_binary_black_hole_parameters(out_sample)
        out_sample = generate_mass_parameters(out_sample)

        return out_sample

    def test_redundancy(self, key, disable_logging=False):
        """
        Test whether adding the key would add be redundant.
        Already existing keys return True.

        Parameters
        ==========
        key: str
            The key to test.
        disable_logging: bool, optional
            Disable logging in this function call. Default is False.

        Returns
        =======
        redundant: bool
            Whether the key is redundant or not
        """
        if key in self:
            logger.debug('{} already in prior'.format(key))
            return True

        sampling_parameters = {key for key in self if not isinstance(
            self[key], (DeltaFunction, Constraint))}

        mass_parameters = {'mass_1', 'mass_2', 'chirp_mass', 'total_mass', 'mass_ratio', 'symmetric_mass_ratio'}
        spin_tilt_1_parameters = {'tilt_1', 'cos_tilt_1'}
        spin_tilt_2_parameters = {'tilt_2', 'cos_tilt_2'}
        spin_azimuth_parameters = {'phi_1', 'phi_2', 'phi_12', 'phi_jl'}
        inclination_parameters = {'theta_jn', 'cos_theta_jn'}
        distance_parameters = {'luminosity_distance', 'comoving_distance', 'redshift'}

        for independent_parameters, parameter_set in \
                zip([2, 2, 1, 1, 1, 1],
                    [mass_parameters, spin_azimuth_parameters,
                     spin_tilt_1_parameters, spin_tilt_2_parameters,
                     inclination_parameters, distance_parameters]):
            if key in parameter_set:
                if len(parameter_set.intersection(
                        sampling_parameters)) >= independent_parameters:
                    logger.disabled = disable_logging
                    logger.warning('{} already in prior. '
                                   'This may lead to unexpected behaviour.'
                                   .format(parameter_set.intersection(self)))
                    logger.disabled = False
                    return True
        return False


class BNSPriorDict(CBCPriorDict):

    def __init__(self, dictionary=None, filename=None, aligned_spin=True,
                 conversion_function=None):
        """ Initialises a Prior set for Binary Neutron Stars

        Parameters
        ==========
        dictionary: dict, optional
            See superclass
        filename: str, optional
            See superclass
        conversion_function: func
            Function to convert between sampled parameters and constraints.
            By default this generates many additional parameters, see
            BNSPriorDict.default_conversion_function
        """
        if aligned_spin:
            default_file = 'aligned_spins_bns_tides_on.prior'
        else:
            default_file = 'precessing_spins_bns_tides_on.prior'
        if dictionary is None and filename is None:
            filename = os.path.join(DEFAULT_PRIOR_DIR, default_file)
            logger.info('No prior given, using default BNS priors in {}.'.format(filename))
        elif filename is not None:
            if not os.path.isfile(filename):
                filename = os.path.join(DEFAULT_PRIOR_DIR, filename)
        super(BNSPriorDict, self).__init__(dictionary=dictionary, filename=filename,
                                           conversion_function=conversion_function)

    def default_conversion_function(self, sample):
        """
        Default parameter conversion function for BNS signals.

        This generates:

        * the parameters passed to source.lal_binary_neutron_star
        * all mass parameters
        * all tidal parameters

        It does not generate:

        * component spins
        * source-frame parameters

        Parameters
        ==========
        sample: dict
            Dictionary to convert

        Returns
        =======
        sample: dict
            Same as input
        """
        out_sample = fill_from_fixed_priors(sample, self)
        out_sample, _ = convert_to_lal_binary_neutron_star_parameters(out_sample)
        out_sample = generate_mass_parameters(out_sample)
        out_sample = generate_tidal_parameters(out_sample)
        return out_sample

    def test_redundancy(self, key, disable_logging=False):
        logger.disabled = disable_logging
        logger.info("Performing redundancy check using BBHPriorDict(self).test_redundancy")
        bbh_redundancy = BBHPriorDict(self).test_redundancy(key)
        logger.disabled = False

        if bbh_redundancy:
            return True
        redundant = False

        sampling_parameters = {key for key in self if not isinstance(
            self[key], (DeltaFunction, Constraint))}

        tidal_parameters = \
            {'lambda_1', 'lambda_2', 'lambda_tilde', 'delta_lambda_tilde'}

        if key in tidal_parameters:
            if len(tidal_parameters.intersection(sampling_parameters)) > 2:
                redundant = True
                logger.disabled = disable_logging
                logger.warning('{} already in prior. '
                               'This may lead to unexpected behaviour.'
                               .format(tidal_parameters.intersection(self)))
                logger.disabled = False
            elif len(tidal_parameters.intersection(sampling_parameters)) == 2:
                redundant = True
        return redundant

    @property
    def tidal(self):
        """ Return true if priors include phase parameters """
        return self.is_nonempty_intersection("tidal")


Prior._default_latex_labels = {
    'mass_1': '$m_1$',
    'mass_2': '$m_2$',
    'total_mass': '$M$',
    'chirp_mass': r'$\mathcal{M}$',
    'mass_ratio': '$q$',
    'symmetric_mass_ratio': r'$\eta$',
    'a_1': '$a_1$',
    'a_2': '$a_2$',
    'tilt_1': r'$\theta_1$',
    'tilt_2': r'$\theta_2$',
    'cos_tilt_1': r'$\cos\theta_1$',
    'cos_tilt_2': r'$\cos\theta_2$',
    'phi_12': r'$\Delta\phi$',
    'phi_jl': r'$\phi_{JL}$',
    'luminosity_distance': '$d_L$',
    'dec': r'$\mathrm{DEC}$',
    'ra': r'$\mathrm{RA}$',
    'iota': r'$\iota$',
    'cos_iota': r'$\cos\iota$',
    'theta_jn': r'$\theta_{JN}$',
    'cos_theta_jn': r'$\cos\theta_{JN}$',
    'psi': r'$\psi$',
    'phase': r'$\phi$',
    'geocent_time': '$t_c$',
    'time_jitter': '$t_j$',
    'lambda_1': r'$\Lambda_1$',
    'lambda_2': r'$\Lambda_2$',
    'lambda_tilde': r'$\tilde{\Lambda}$',
    'delta_lambda_tilde': r'$\delta\tilde{\Lambda}$',
    'chi_1': r'$\chi_1$',
    'chi_2': r'$\chi_2$',
    'chi_1_in_plane': r'$\chi_{1, \perp}$',
    'chi_2_in_plane': r'$\chi_{2, \perp}$',
}


class CalibrationPriorDict(PriorDict):
    """
    Prior dictionary class for calibration parameters.
    This has methods for simplifying the creation of priors for the large
    numbers of parameters used with the spline model.
    """

    def __init__(self, dictionary=None, filename=None):
        """
        Initialises a Prior dictionary for calibration parameters

        Parameters
        ==========
        dictionary: dict, optional
            See superclass
        filename: str, optional
            See superclass
        """
        if dictionary is None and filename is not None:
            filename = os.path.join(DEFAULT_PRIOR_DIR, filename)
        super(CalibrationPriorDict, self).__init__(dictionary=dictionary, filename=filename)
        self.source = None

    def to_file(self, outdir, label):
        """
        Write the prior to file. This includes information about the source if
        possible.

        Parameters
        ==========
        outdir: str
            Output directory.
        label: str
            Label for prior.
        """
        PriorDict.to_file(self, outdir=outdir, label=label)
        if self.source is not None:
            prior_file = os.path.join(outdir, "{}.prior".format(label))
            with open(prior_file, "a") as outfile:
                outfile.write("# prior source file is {}".format(self.source))

    @staticmethod
    def from_envelope_file(envelope_file, minimum_frequency,
                           maximum_frequency, n_nodes, label,
                           boundary="reflective", correction_type=None):
        """
        Load in the calibration envelope.

        This is a text file with columns

        ::

            frequency median-amplitude median-phase -1-sigma-amplitude
            -1-sigma-phase +1-sigma-amplitude +1-sigma-phase

        There are two definitions of the calibration correction in the
        literature, one defines the correction as mapping calibrated strain
        to theoretical waveform templates (:code:`data`) and the other as
        mapping theoretical waveform templates to calibrated strain
        (:code:`template`). Prior to version 1.4.0, :code:`template` was assumed,
        the default changed to :code:`data` when the :code:`correction` argument
        was added.

        Parameters
        ==========
        envelope_file: str
            Name of file to read in.
        minimum_frequency: float
            Minimum frequency for the spline.
        maximum_frequency: float
            Minimum frequency for the spline.
        n_nodes: int
            Number of nodes for the spline.
        label: str
            Label for the names of the parameters, e.g., `recalib_H1_`
        boundary: None, 'reflective', 'periodic'
            The type of prior boundary to assign
        correction_type: str
            How the correction is defined, either to the :code:`data`
            (default) or the :code:`template`. In general, data products
            produced by the LVK calibration groups assume :code:`data`.
            The default value will be removed in a future release and
            this will need to be explicitly specified.

            .. versionadded:: 1.4.0

        Returns
        =======
        prior: PriorDict
            Priors for the relevant parameters.
            This includes the frequencies of the nodes which are _not_ sampled.
        """
        from .detector.calibration import _check_calibration_correction_type
        correction_type = _check_calibration_correction_type(correction_type=correction_type)

        calibration_data = np.genfromtxt(envelope_file).T
        log_frequency_array = np.log(calibration_data[0])

        if correction_type.lower() == "data":
            amplitude_median = 1 / calibration_data[1] - 1
            phase_median = -calibration_data[2]
            amplitude_sigma = abs(1 / calibration_data[3] - 1 / calibration_data[5]) / 2
            phase_sigma = abs(calibration_data[6] - calibration_data[4]) / 2
        else:
            amplitude_median = calibration_data[1] - 1
            phase_median = calibration_data[2]
            amplitude_sigma = abs(calibration_data[5] - calibration_data[3]) / 2
            phase_sigma = abs(calibration_data[6] - calibration_data[4]) / 2

        log_nodes = np.linspace(np.log(minimum_frequency),
                                np.log(maximum_frequency), n_nodes)

        amplitude_mean_nodes = \
            InterpolatedUnivariateSpline(log_frequency_array, amplitude_median)(log_nodes)
        amplitude_sigma_nodes = \
            InterpolatedUnivariateSpline(log_frequency_array, amplitude_sigma)(log_nodes)
        phase_mean_nodes = \
            InterpolatedUnivariateSpline(log_frequency_array, phase_median)(log_nodes)
        phase_sigma_nodes = \
            InterpolatedUnivariateSpline(log_frequency_array, phase_sigma)(log_nodes)

        prior = CalibrationPriorDict()
        for ii in range(n_nodes):
            name = "recalib_{}_amplitude_{}".format(label, ii)
            latex_label = "$A^{}_{}$".format(label, ii)
            prior[name] = Gaussian(mu=amplitude_mean_nodes[ii],
                                   sigma=amplitude_sigma_nodes[ii],
                                   name=name, latex_label=latex_label,
                                   boundary=boundary)
        for ii in range(n_nodes):
            name = "recalib_{}_phase_{}".format(label, ii)
            latex_label = r"$\phi^{}_{}$".format(label, ii)
            prior[name] = Gaussian(mu=phase_mean_nodes[ii],
                                   sigma=phase_sigma_nodes[ii],
                                   name=name, latex_label=latex_label,
                                   boundary=boundary)
        for ii in range(n_nodes):
            name = "recalib_{}_frequency_{}".format(label, ii)
            latex_label = "$f^{}_{}$".format(label, ii)
            prior[name] = DeltaFunction(peak=np.exp(log_nodes[ii]), name=name,
                                        latex_label=latex_label)
        prior.source = os.path.abspath(envelope_file)
        return prior

    @staticmethod
    def constant_uncertainty_spline(
            amplitude_sigma, phase_sigma, minimum_frequency, maximum_frequency,
            n_nodes, label, boundary="reflective"):
        """
        Make prior assuming constant in frequency calibration uncertainty.

        This assumes Gaussian fluctuations about 0.

        Parameters
        ==========
        amplitude_sigma: float
            Uncertainty in the amplitude.
        phase_sigma: float
            Uncertainty in the phase.
        minimum_frequency: float
            Minimum frequency for the spline.
        maximum_frequency: float
            Minimum frequency for the spline.
        n_nodes: int
            Number of nodes for the spline.
        label: str
            Label for the names of the parameters, e.g., `recalib_H1_`
        boundary: None, 'reflective', 'periodic'
            The type of prior boundary to assign

        Returns
        =======
        prior: PriorDict
            Priors for the relevant parameters.
            This includes the frequencies of the nodes which are _not_ sampled.
        """
        nodes = np.logspace(np.log10(minimum_frequency),
                            np.log10(maximum_frequency), n_nodes)

        amplitude_mean_nodes = [0] * n_nodes
        amplitude_sigma_nodes = [amplitude_sigma] * n_nodes
        phase_mean_nodes = [0] * n_nodes
        phase_sigma_nodes = [phase_sigma] * n_nodes

        prior = CalibrationPriorDict()
        for ii in range(n_nodes):
            name = "recalib_{}_amplitude_{}".format(label, ii)
            latex_label = "$A^{}_{}$".format(label, ii)
            prior[name] = Gaussian(mu=amplitude_mean_nodes[ii],
                                   sigma=amplitude_sigma_nodes[ii],
                                   name=name, latex_label=latex_label,
                                   boundary=boundary)
        for ii in range(n_nodes):
            name = "recalib_{}_phase_{}".format(label, ii)
            latex_label = r"$\phi^{}_{}$".format(label, ii)
            prior[name] = Gaussian(mu=phase_mean_nodes[ii],
                                   sigma=phase_sigma_nodes[ii],
                                   name=name, latex_label=latex_label,
                                   boundary=boundary)
        for ii in range(n_nodes):
            name = "recalib_{}_frequency_{}".format(label, ii)
            latex_label = "$f^{}_{}$".format(label, ii)
            prior[name] = DeltaFunction(peak=nodes[ii], name=name,
                                        latex_label=latex_label)

        return prior


def secondary_mass_condition_function(reference_params, mass_1):
    """
    Condition function to use for a prior that is conditional on the value
    of the primary mass, for example, a prior on the secondary mass that is
    bounded by the primary mass.

    .. code-block:: python

        import bilby
        priors = bilby.gw.prior.CBCPriorDict()
        priors["mass_1"] = bilby.core.prior.Uniform(5, 50)
        priors["mass_2"] = bilby.core.prior.ConditionalUniform(
            condition_func=bilby.gw.prior.secondary_mass_condition_function,
            minimum=5,
            maximum=50,
        )

    Parameters
    ----------
    reference_params: dict
        Reference parameter dictionary, this should contain a "minimum"
        attribute.
    mass_1: float
        The primary mass to use as the upper limit for the prior.

    Returns
    -------
    dict:
        Updated prior limits given the provided primary mass.
    """
    return dict(minimum=reference_params['minimum'], maximum=mass_1)


ConditionalCosmological = conditional_prior_factory(Cosmological)
ConditionalUniformComovingVolume = conditional_prior_factory(UniformComovingVolume)
ConditionalUniformSourceFrame = conditional_prior_factory(UniformSourceFrame)


class HealPixMapPriorDist(BaseJointPriorDist):
    """
    Class defining prior according to given HealPix Map, defaults to 2D in ra and dec but can be set to include
    Distance as well. This only works with skymaps that include the 2D joint probability in ra/dec and that use the
    normal LALInference type skymaps where each pixel has a DISTMU, DISTSIGMA, and DISTNORM defining the conditional
    distance distribution along a given line of sight.

    Parameters
    ==========

    hp_file : file path to .fits file
        .fits file that contains the 2D or 3D Healpix Map
    names : list (optional)
        list of names of parameters included in the JointPriorDist, defaults to ['ra', 'dec']
    bounds : dict or list (optional)
        dictionary or list with given prior bounds. defaults to normal bounds on ra, dev and 0, inf for distance
        if this is for a 3D map

    Returns
    =======

    PriorDist : `bilby.gw.prior.HealPixMapPriorDist`
        A JointPriorDist object to store the joint prior distribution according to passed healpix map
    """
    def __init__(self, hp_file, names=None, bounds=None, distance=False):
        self.hp = self._check_imports()
        self.hp_file = hp_file
        if names is None:
            names = ["ra", "dec"]
        if bounds is None:
            bounds = [[0, 2 * np.pi], [-np.pi / 2.0, np.pi / 2.0]]
        elif isinstance(bounds, dict):
            bs = [[] for _ in bounds.keys()]
            for i, key in enumerate(bounds.keys()):
                bs[i] = (bounds[key][0], bounds[key][1])
            bounds = bs
        if distance:
            if len(names) == 2:
                names.append("distance")
            if len(bounds) == 2:
                bounds.append([0, np.inf])
            self.distance = True
            self.prob, self.distmu, self.distsigma, self.distnorm = self.hp.read_map(
                hp_file, field=range(4)
            )
        else:
            self.distance = False
            self.prob = self.hp.read_map(hp_file)
        self.prob = self._check_norm(self.prob)

        super(HealPixMapPriorDist, self).__init__(names=names, bounds=bounds)
        self.distname = "hpmap"
        self.npix = len(self.prob)
        self.nside = self.hp.npix2nside(self.npix)
        self.pixel_area = self.hp.nside2pixarea(self.nside)
        self.pixel_length = self.pixel_area ** (1 / 2.0)
        self.pix_xx = np.arange(self.npix)
        self._all_interped = interp1d(x=self.pix_xx, y=self.prob, bounds_error=False, fill_value=0)
        self.inverse_cdf = None
        self.distance_pdf = None
        self.distance_icdf = None
        self._build_attributes()
        name = self.names[-1]
        if self.bounds[name][1] != np.inf and self.bounds[name][0] != -np.inf:
            self.rs = np.linspace(self.bounds[name][0], self.bounds[name][1], 1000)
        else:
            self.rs = np.linspace(0, 5000, 1000)

    def _build_attributes(self):
        """
        Method that builds the inverse cdf of the P(pixel) distribution for rescaling
        """
        yy = self._all_interped(self.pix_xx)
        yy /= trapezoid(yy, self.pix_xx)
        YY = cumulative_trapezoid(yy, self.pix_xx, initial=0)
        YY[-1] = 1
        self.inverse_cdf = interp1d(x=YY, y=self.pix_xx, bounds_error=True)

    @staticmethod
    def _check_imports():
        """
        Static method to check that healpy is installed on the machine running bibly
        """
        try:
            import healpy
        except Exception:
            raise ImportError("Must have healpy installed on this machine to use HealPixMapPrior")
        return healpy

    def _rescale(self, samp, **kwargs):
        """
        Overwrites the _rescale method of BaseJoint Prior to rescale a single value from the unitcube onto
        two values (ra, dec) or 3 (ra, dec, dist) if distance is included

        Parameters
        ==========
        samp : float, int
            must take in single value for pixel on unitcube to recale onto ra, dec (distance), for the map Prior
        kwargs : dict
            kwargs are all passed to _rescale() method

        Returns
        =======
        rescaled_sample : array_like
            sample to rescale onto the prior
        """
        if self.distance:
            dist_samp = samp[:, -1]
            samp = samp[:, 0]
        else:
            samp = samp[:, 0]
        pix_rescale = self.inverse_cdf(samp)
        sample = np.empty((len(pix_rescale), 2))
        dist_samples = np.empty((len(pix_rescale)))
        for i, val in enumerate(pix_rescale):
            theta, ra = self.hp.pix2ang(self.nside, int(round(val)))
            dec = 0.5 * np.pi - theta
            sample[i, :] = self.draw_from_pixel(ra, dec, int(round(val)))
            if self.distance:
                self.update_distance(int(round(val)))
                dist_samples[i] = self.distance_icdf(dist_samp[i])
        if self.distance:
            sample = np.vstack([sample[:, 0], sample[:, 1], dist_samples])
        return sample.reshape((-1, self.num_vars))

    def update_distance(self, pix_idx):
        """
        Method to update the conditional distance distributions at given pixel used for distance handling in the
        JointPrior Parameters. This function updates the current distance pdf, inverse_cdf, and sampler according to
        given pixel or line of sight.

        Parameters
        ==========
        pix_idx : int
            pixel index value to create the distribution for

        Returns
        =======
        None : None
            just updates these functions at new pixel values
        """
        self.distance_pdf = lambda r: self.distnorm[pix_idx] * norm(
            loc=self.distmu[pix_idx], scale=self.distsigma[pix_idx]
        ).pdf(r)
        pdfs = self.rs ** 2 * self.distance_pdf(self.rs)
        cdfs = np.cumsum(pdfs) / np.sum(pdfs)
        self.distance_icdf = interp1d(cdfs, self.rs)

    @staticmethod
    def _check_norm(array):
        """
        static method to check if array is properlly normalized and if not to normalize it.

        Parameters
        ==========
        array : array_like
            input array we want to renormalize if not already normalized

        Returns
        =======
        normed_array : array_like
            returns input array normalized
        """
        norm = np.linalg.norm(array, ord=1)
        if norm == 0:
            norm = np.finfo(array.dtype).eps
        return array / norm

    def _sample(self, size, **kwargs):
        """
        Overwrites the _sample method of BaseJoint Prior. Picks a pixel value according to their probabilities, then
        uniformly samples ra, and decs that are contained in chosen pixel. If the PriorDist includes distance it then
        updates the distance distributions and will sample according to the conditional distance distribution along a
        given line of sight

        Parameters
        ==========
        size : int
            number of samples we want to draw
        kwargs : dict
            kwargs are all passed to be used

        Returns
        =======
        sample : array_like
            sample of ra, and dec (and distance if 3D=True)
        """
        sample_pix = random.rng.choice(self.npix, size=size, p=self.prob, replace=True)
        sample = np.empty((size, self.num_vars))
        for samp in range(size):
            theta, ra = self.hp.pix2ang(self.nside, sample_pix[samp])
            dec = 0.5 * np.pi - theta
            if self.distance:
                self.update_distance(sample_pix[samp])
                dist = self.draw_distance(sample_pix[samp])
                ra_dec = self.draw_from_pixel(ra, dec, sample_pix[samp])
                sample[samp, :] = [ra_dec[0], ra_dec[1], dist]
            else:
                sample[samp, :] = self.draw_from_pixel(ra, dec, sample_pix[samp])
        return sample.reshape((-1, self.num_vars))

    def draw_distance(self, pix):
        """
        Method to recursively draw a distance value from the given set distance distribution and check that it is in
        the bounds

        Parameters
        ==========

        pix : int
            integer for pixel to draw a distance from

        Returns
        =======
        dist : float
            sample drawn from the distance distribution at set pixel index
        """
        if self.distmu[pix] == np.inf or self.distmu[pix] <= 0:
            return 0
        dist = self.distance_icdf(random.rng.uniform(0, 1))
        name = self.names[-1]
        if (dist > self.bounds[name][1]) | (dist < self.bounds[name][0]):
            self.draw_distance(pix)
        else:
            return dist

    def draw_from_pixel(self, ra, dec, pix):
        """
        Recursive function to uniformly draw ra, and dec values that are located in the given pixel

        Parameters
        ==========
        ra : float, int
            value drawn for rightascension
        dec : float, int
            value drawn for declination
        pix : int
            pixel index for given pixel we want to get ra, and dec from

        Returns
        =======
        ra_dec : tuple
            this returns a tuple of ra, and dec sampled uniformly that are in the pixel given
        """
        if not self.check_in_pixel(ra, dec, pix):
            self.draw_from_pixel(ra, dec, pix)
        return np.array(
            [
                random.rng.uniform(ra - self.pixel_length, ra + self.pixel_length),
                random.rng.uniform(dec - self.pixel_length, dec + self.pixel_length),
            ]
        )

    def check_in_pixel(self, ra, dec, pix):
        """
        Method that checks if given rightacension and declination values are within the given pixel index and the bounds

        Parameters
        ==========
        ra : float, int
            rightascension value to check
        dec : float, int
            declination value to check
        pix : int
            index for pixel we want to check in

        Returns
        =======
        bool :
            returns True if values inside pixel, False if not
        """
        for val, name in zip([ra, dec], self.names):
            if (val < self.bounds[name][0]) or (val > self.bounds[name][1]):
                return False
        phi, theta = ra, 0.5 * np.pi - dec
        pixel = self.hp.ang2pix(self.nside, theta, phi)
        return pix == pixel

    def _ln_prob(self, samp, lnprob, outbounds):
        """
        Overwrites the _lnprob method of BaseJoint Prior

        Parameters
        ==========
        samp : array_like
            samples of ra, dec to evaluate the lnprob at
        lnprob : array_like
            array of correct length we want to populate with lnprob values
        outbounds : boolean array
            boolean array that flags samples that are out of the given bounds

        Returns
        =======
        lnprob : array_like
            lnprob values at each sample
        """
        for i in range(samp.shape[0]):
            if not outbounds[i]:
                if self.distance:
                    phi, dec, dist = samp[0]
                else:
                    phi, dec = samp[0]
                theta = 0.5 * np.pi - dec
                pixel = self.hp.ang2pix(self.nside, theta, phi)
                lnprob[i] = np.log(self.prob[pixel] / self.pixel_area)
                if self.distance:
                    self.update_distance(pixel)
                    lnprob[i] += np.log(self.distance_pdf(dist) * dist ** 2)
        lnprob[outbounds] = -np.inf
        return lnprob

    def __eq__(self, other):
        skip_keys = ["_all_interped", "inverse_cdf", "distance_pdf", "distance_icdf"]
        if self.__class__ != other.__class__:
            return False
        if sorted(self.__dict__.keys()) != sorted(other.__dict__.keys()):
            return False
        for key in self.__dict__:
            if key in skip_keys:
                continue
            if key == "hp_file":
                if self.__dict__[key] != other.__dict__[key]:
                    return False
            elif isinstance(self.__dict__[key], (np.ndarray, list)):
                thisarr = np.asarray(self.__dict__[key])
                otherarr = np.asarray(other.__dict__[key])
                if thisarr.dtype == float and otherarr.dtype == float:
                    fin1 = np.isfinite(np.asarray(self.__dict__[key]))
                    fin2 = np.isfinite(np.asarray(other.__dict__[key]))
                    if not np.array_equal(fin1, fin2):
                        return False
                    if not np.allclose(thisarr[fin1], otherarr[fin2], atol=1e-15):
                        return False
                else:
                    if not np.array_equal(thisarr, otherarr):
                        return False
            else:
                if not self.__dict__[key] == other.__dict__[key]:
                    return False
        return True


class HealPixPrior(JointPrior):
    """
    A prior distribution that follows a user-provided HealPix map for one
    parameter.

    See :code:`bilby.gw.prior.HealPixMapPriorDist` for more details of how to
    instantiate the prior.
    """
    def __init__(self, dist, name=None, latex_label=None, unit=None):
        """

        Parameters
        ----------
        dist: bilby.gw.prior.HealPixMapPriorDist
            The base joint probability.
        name: str
            The name of the parameter, it should be contained in the map.
            One of ["ra", "dec", "luminosity_distance"].
        latex_label: str
            Latex label used for plotting, will be read from default values if
            not provided.
        unit: str
            The unit of the parameter.
        """
        if not isinstance(dist, HealPixMapPriorDist):
            raise JointPriorDistError("dist object must be instance of HealPixMapPriorDist")
        super(HealPixPrior, self).__init__(dist=dist, name=name, latex_label=latex_label, unit=unit)


class TOVJointDist(BaseJointPriorDist):
    """
    Joint (mass, tidal deformability) distribution built from a gridded 2D
    KDE of TOV mass-Lambda sequences across many equations of state.

    Reproduces the correlation between a neutron star's mass and its tidal
    deformability implied by the EOS population, rather than treating them
    as independent 1D priors. The grid itself is produced offline (fitting
    the KDE directly to the full population of TOV mass-Lambda samples is
    too expensive to redo per job/per likelihood call) and loaded here from
    an ``.npz`` file.
    """

    def __init__(self, grid_file, names, bounds=None):
        """
        Parameters
        ==========
        grid_file: str
            Path to an .npz file with arrays ``mass_edges`` (N,),
            ``log10_lambda_edges`` (M,), and ``log_pdf`` (N, M) -- the
            natural log of p(mass, log10(Lambda)) on that grid.
        names: list
            Exactly two parameter names, [mass_name, lambda_name], e.g.
            ``["mass_1", "lambda_1"]``.
        bounds: list, optional
            Bounds for [mass_name, lambda_name]. Defaults to the grid's own
            (mass, Lambda) extent.
        """
        if len(names) != 2:
            raise ValueError(
                "TOVJointDist needs exactly two parameter names: [mass, lambda]"
            )

        self.grid_file = grid_file
        data = np.load(grid_file)
        self.mass_edges = data["mass_edges"]
        self.log10_lambda_edges = data["log10_lambda_edges"]
        log_pdf = data["log_pdf"]

        if bounds is None:
            lambda_edges = 10 ** self.log10_lambda_edges
            bounds = [
                (float(self.mass_edges[0]), float(self.mass_edges[-1])),
                (float(lambda_edges[0]), float(lambda_edges[-1])),
            ]
        super(TOVJointDist, self).__init__(names=names, bounds=bounds)
        # ConditionalPriorDict.rescale groups sibling JointPriors by this
        # string (see bilby/core/prior/dict.py's `joint` dict), not by
        # `dist` object identity -- must be unique per TOVJointDist instance
        # (e.g. the mass_1/lambda_1 pair vs. the mass_2/lambda_2 pair) or
        # rescale results from unrelated instances get merged together.
        self.distname = "tov_joint_" + "_".join(names)

        pdf = np.exp(log_pdf)
        # Raw table, kept for TOVConditionalLambdaPrior, which integrates the
        # interpolant below analytically along log10(Lambda).
        self._log_pdf = log_pdf

        # Interpolant for ln p(mass, log10(Lambda)) itself, used by _ln_prob.
        self._log_pdf_interp = RegularGridInterpolator(
            (self.mass_edges, self.log10_lambda_edges), log_pdf,
            method="linear", bounds_error=False, fill_value=-np.inf,
        )

        # p(mass): integrate the joint density over log10(Lambda), then its
        # inverse CDF, used to rescale the mass unit-cube coordinate.
        mass_marginal = np.clip(trapezoid(pdf, self.log10_lambda_edges, axis=1), 0, None)
        self._mass_cdf = cumulative_trapezoid(mass_marginal, self.mass_edges, initial=0)
        self._mass_cdf /= self._mass_cdf[-1]

        # p(log10(Lambda) | mass) inverse CDF, one row per mass grid point,
        # used to rescale the lambda unit-cube coordinate conditional on the
        # mass value just drawn.
        row_sums = trapezoid(pdf, self.log10_lambda_edges, axis=1)[:, None]
        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        cond_pdf = pdf / row_sums
        cond_cdf = cumulative_trapezoid(cond_pdf, self.log10_lambda_edges, axis=1, initial=0)
        cond_cdf /= np.clip(cond_cdf[:, -1:], 1e-300, None)
        self._cond_cdf = cond_cdf  # shape (N_mass, N_lambda)

    def mass_cdf(self, mass):
        """CDF of the marginal mass distribution p(mass).

        Parameters
        ==========
        mass: array_like
            Mass values, in the same units as the grid's ``mass_edges``.

        Returns
        =======
        array_like: F(mass), clipped to [0, 1] outside the grid.
        """
        return np.interp(mass, self.mass_edges, self._mass_cdf)

    def mass_from_cdf(self, u):
        """Inverse of :meth:`mass_cdf`, i.e. the mass unit-cube rescaling."""
        return np.interp(u, self._mass_cdf, self.mass_edges)

    def log10_lambda_from_cdf(self, u, mass):
        """Inverse CDF of p(log10(Lambda) | mass).

        The conditional CDF is tabulated per mass grid point; rows are linearly
        interpolated in mass before inverting.

        Parameters
        ==========
        u: array_like
            Unit-cube coordinates.
        mass: array_like
            Masses to condition on, same shape as :code:`u`.

        Returns
        =======
        array_like: log10(Lambda) values.
        """
        u = np.atleast_1d(u)
        mass = np.atleast_1d(mass)

        row_idx = np.clip(np.searchsorted(self.mass_edges, mass), 1, len(self.mass_edges) - 1)
        i0, i1 = row_idx - 1, row_idx
        span = self.mass_edges[i1] - self.mass_edges[i0]
        safe_span = np.where(span > 0, span, 1.0)
        w = np.where(span > 0, (mass - self.mass_edges[i0]) / safe_span, 0.0)

        log10_lambda = np.empty_like(mass, dtype=float)
        for k in range(len(mass)):
            cdf_row = (1.0 - w[k]) * self._cond_cdf[i0[k]] + w[k] * self._cond_cdf[i1[k]]
            log10_lambda[k] = np.interp(u[k], cdf_row, self.log10_lambda_edges)
        return log10_lambda

    def ln_pdf_mass_lambda(self, mass, lam):
        """ln p(mass, Lambda) from the gridded KDE.

        The grid holds p(mass, log10(Lambda)); the Jacobian of
        log10(Lambda) -> Lambda is applied here, so this is a density in
        Lambda. Returns -inf off the grid.
        """
        mass = np.atleast_1d(mass)
        lam = np.atleast_1d(lam)
        log10_lam = np.log10(np.clip(lam, 1e-300, None))
        log_pdf_val = self._log_pdf_interp(np.column_stack([mass, log10_lam]))
        # p(mass, Lambda) = p(mass, log10 Lambda) / (Lambda * ln(10)),
        # so the log Jacobian is -log(Lambda) - log(log(10)). NB the second
        # term is log(log(10)) = 0.834, not log(10) = 2.303: an earlier version
        # of this had log(10), which left ln_prob low by a constant
        # 10 / ln(10) = 4.34 per (mass, Lambda) pair. Being constant it did not
        # bias posteriors, but it did shift the log evidence and made the
        # density unnormalised.
        return log_pdf_val - np.log(lam) - np.log(np.log(10.0))

    def _rescale(self, samp, **kwargs):
        mass = self.mass_from_cdf(samp[:, 0])
        lam = 10 ** self.log10_lambda_from_cdf(samp[:, 1], mass)
        return np.column_stack([mass, lam])

    def _sample(self, size, *, random_state=None, **kwargs):
        # BaseJointPriorDist.sample passes random_state through; honour it so
        # seeded runs are reproducible and pool workers get distinct streams.
        rng = random.resolve_random_state(random_state)
        u = rng.uniform(0, 1, size=(size, 2))
        return self._rescale(u)

    @xp_wrap
    def _ln_prob(self, samp, lnprob, outbounds, *, xp=None):
        # `xp` is supplied by BaseJointPriorDist.ln_prob. The KDE is evaluated
        # with a scipy RegularGridInterpolator, which is numpy-only, so the
        # interpolation is done in numpy and converted back on the way out.
        # This makes the class numpy-bound: an xp of cupy/jax would fail here.
        samp = np.asarray(samp)
        lnprob_vals = self.ln_pdf_mass_lambda(samp[:, 0], samp[:, 1])
        return xp.where(xp.asarray(outbounds), -np.inf, xp.asarray(lnprob_vals))

    @xp_wrap
    def ln_prob(self, value, *, xp=None):
        """Log-probability of a sample, as a scalar for single samples.

        BaseJointPriorDist.ln_prob ends in ``return lnprob[()]``, which unwraps
        0-d arrays but leaves a single-sample result at shape (1,). Ordinary
        priors return floats, so PriorDict.ln_prob's ``xp.asarray([...])`` over
        the whole prior set then raises

            ValueError: setting an array element with a sequence. The requested
            array has an inhomogeneous shape after 1 dimensions.

        ``rescale`` already squeezes the equivalent case (see
        BaseJointPriorDist.rescale); this brings ln_prob into line. Kept local
        to TOVJointDist rather than fixed in the base class so this fork stays a
        clean addition on top of upstream. Harmless if upstream is fixed later.
        """
        lnprob = super().ln_prob(value, xp=xp)
        if getattr(lnprob, "shape", None) == (1,):
            lnprob = lnprob[0]
        return lnprob


class TOVJointPrior(JointPrior):
    """
    A prior distribution that follows a TOV-informed joint (mass, tidal
    deformability) KDE for one parameter.

    See :code:`bilby.gw.prior.TOVJointDist` for how to instantiate the
    shared distribution object.
    """

    def __init__(self, dist, name=None, latex_label=None, unit=None):
        """
        Parameters
        ----------
        dist: bilby.gw.prior.TOVJointDist
            The base joint probability.
        name: str
            The name of the parameter; one of dist.names (e.g. "mass_1" or
            "lambda_1").
        latex_label: str
            Latex label used for plotting.
        unit: str
            The unit of the parameter.
        """
        if not isinstance(dist, TOVJointDist):
            raise JointPriorDistError("dist object must be instance of TOVJointDist")
        super(TOVJointPrior, self).__init__(dist=dist, name=name, latex_label=latex_label, unit=unit)

    def sample(self, size=1, *, random_state=None, **kwargs):
        """Draw a sample, returning a scalar rather than a 0-d array.

        JointPrior.sample ends in ``return sample.squeeze()``. For a size-1 draw
        squeeze yields ``array(1.46)`` - a 0-d ndarray, not a float. That then
        reaches lalsimulation's SimInspiralWaveformParamsInsertTidalLambda1,
        whose SWIG REAL8 typemap rejects arrays of any shape:

            TypeError: in method 'SimInspiralWaveformParamsInsertTidalLambda1',
                       argument 2 of type 'REAL8'

        Indexing with ``[()]`` unwraps 0-d arrays to numpy scalars and leaves
        larger results untouched - the idiom BaseJointPriorDist.sample already
        uses. It is also a no-op on numpy scalars, so this stays correct if
        upstream is fixed later.

        Note the masses only avoided this by luck: they are multiplied by
        lal.MSUN_SI before reaching LAL, which collapses them to numpy scalars.
        """
        return super().sample(size=size, random_state=random_state, **kwargs)[()]


class TOVConditionalLambdaPrior(ConditionalBasePrior):
    """
    Conditional prior p(Lambda | mass) on one star's tidal deformability,
    from the TOV mass-Lambda KDE, for use with an independent mass prior.

    Where :code:`TOVJointDist` draws mass and Lambda together, this prior
    takes the star's mass from whatever mass prior the run already has and
    puts the EOS conditional on Lambda alone:

        p(Lambda | m) = p_TOV(m, Lambda) / p_TOV(m)

    The density is the same bilinear-in-ln(p) interpolant that
    :code:`TOVJointDist.ln_pdf_mass_lambda` evaluates. At fixed mass that
    interpolant is piecewise linear in log10(Lambda), so the normalisation
    p_TOV(m) and the inverse CDF used by :meth:`rescale` are computed exactly
    from it, cell by cell, rather than from separately tabulated CDFs.
    :meth:`rescale` therefore samples exactly the density :meth:`ln_prob`
    returns, and that density integrates to one at every mass.

    Masses outside the grid have no conditional: :meth:`ln_prob` returns
    -inf there and :meth:`rescale` returns nan, so the mass prior should stay
    inside the grid, or be cut to it with a :code:`Constraint`.

    Must be used in a :code:`ConditionalPriorDict` (e.g.
    :code:`BNSPriorDict`), which supplies the masses, e.g.

    .. code-block:: python

        priors["mass_1"] = Uniform(1.0, 2.0, "mass_1")
        priors["mass_2"] = Uniform(1.0, 2.0, "mass_2")
        priors["lambda_1"] = TOVConditionalLambdaPrior(grid_file, component=1)
        priors["lambda_2"] = TOVConditionalLambdaPrior(grid_file, component=2)
    """

    _mass_parameter_options = ("component_masses", "chirp_mass_and_mass_ratio")

    def __init__(self, grid_file, component, mass_parameters="component_masses",
                 name=None, latex_label=None, unit=None, boundary=None):
        """
        Parameters
        ==========
        grid_file: str or TOVJointDist
            Path to the .npz grid used by :code:`TOVJointDist`, or an existing
            :code:`TOVJointDist` to share its loaded grid.
        component: int
            1 or 2: which star this Lambda belongs to.
        mass_parameters: str
            Which sampled parameters the mass is read from.
            ``"component_masses"`` (default) conditions on
            ``mass_<component>``. ``"chirp_mass_and_mass_ratio"`` conditions on
            ``chirp_mass`` and ``mass_ratio`` and converts them to the
            component mass, for runs that sample those and constrain the
            component masses.
        name: str, optional
            Defaults to ``lambda_<component>``.
        latex_label, unit, boundary:
            See superclass.
        """
        if component not in (1, 2):
            raise ValueError("component must be 1 or 2")
        if mass_parameters not in self._mass_parameter_options:
            raise ValueError(
                f"mass_parameters must be one of {self._mass_parameter_options}"
            )
        if isinstance(grid_file, TOVJointDist):
            self._grid = grid_file
            grid_file = grid_file.grid_file
        else:
            # The names are internal; this dist is only used for its grid.
            self._grid = TOVJointDist(grid_file, names=["mass", "lambda"])
        self.grid_file = grid_file
        self.component = component
        self.mass_parameters = mass_parameters
        if name is None:
            name = f"lambda_{component}"

        super(TOVConditionalLambdaPrior, self).__init__(
            name=name, latex_label=latex_label, unit=unit, boundary=boundary,
            condition_func=self._condition_function,
            minimum=float(10 ** self._grid.log10_lambda_edges[0]),
            maximum=float(10 ** self._grid.log10_lambda_edges[-1]),
        )
        if mass_parameters == "component_masses":
            self._required_variables = [f"mass_{component}"]
        else:
            self._required_variables = ["chirp_mass", "mass_ratio"]
        self._warned_off_grid = False
        # conditional_prior_factory renames the class on instantiation; undo
        # that, as ConditionalChiInPlane does, so repr and JSON round-trip.
        self.__class__.__name__ = "TOVConditionalLambdaPrior"
        self.__class__.__qualname__ = "TOVConditionalLambdaPrior"

    def _condition_function(self, reference_params, **kwargs):
        # The conditioning is done in the methods below from the masses
        # themselves; no attribute of the prior changes with the mass.
        return dict()

    def __repr__(self):
        return Prior.__repr__(self)

    def get_instantiation_dict(self):
        return Prior.get_instantiation_dict(self)

    def _mass(self, required_variables):
        """The conditioning mass, from the required variables."""
        required_variables.pop("xp", None)
        if not required_variables:
            raise ValueError(
                f"{self.__class__.__name__} needs {self.required_variables}; use it "
                "in a ConditionalPriorDict (e.g. BNSPriorDict), which supplies them."
            )
        self.update_conditions(**required_variables)
        if self.mass_parameters == "component_masses":
            return np.asarray(required_variables[f"mass_{self.component}"], dtype=float)
        chirp_mass = np.asarray(required_variables["chirp_mass"], dtype=float)
        mass_ratio = np.asarray(required_variables["mass_ratio"], dtype=float)
        total_mass = chirp_mass_and_mass_ratio_to_total_mass(chirp_mass, mass_ratio)
        masses = total_mass_and_mass_ratio_to_component_masses(mass_ratio, total_mass)
        return np.asarray(masses[self.component - 1], dtype=float)

    def _conditional_tables(self, mass):
        """Per-cell quantities of the interpolant along log10(Lambda) at each mass.

        At fixed mass the bilinear interpolant of ln p(mass, log10 Lambda) is
        piecewise linear in log10 Lambda, with node values a_j blended
        linearly in mass. On cell j, of width h_j and slope d_j = a_{j+1} - a_j,
        the density is exp(a_j + d_j t) for t in [0, 1], so its integral is
        h_j (e^{a_{j+1}} - e^{a_j}) / d_j. Everything is shifted by the row
        maximum so the exponentials neither overflow nor all underflow.

        Returns
        =======
        valid: mass inside the grid, shape (n,)
        a: shifted node values, shape (n, M)
        cum: cumulative cell integrals from 0, shape (n, M)
        ln_norm: ln of the unshifted total integral, i.e. ln p_TOV(mass)
        """
        grid = self._grid
        edges = grid.mass_edges
        valid = (mass >= edges[0]) & (mass <= edges[-1])
        safe_mass = np.where(valid, mass, edges[0])
        row_idx = np.clip(np.searchsorted(edges, safe_mass), 1, len(edges) - 1)
        i0, i1 = row_idx - 1, row_idx
        w = (safe_mass - edges[i0]) / (edges[i1] - edges[i0])
        a = (1 - w)[:, None] * grid._log_pdf[i0] + w[:, None] * grid._log_pdf[i1]
        shift = a.max(axis=1)
        a = a - shift[:, None]
        e = np.exp(a)
        d = np.diff(a, axis=1)
        h = np.diff(grid.log10_lambda_edges)
        small = np.abs(d) < 1e-8
        with np.errstate(divide="ignore", invalid="ignore"):
            cell = h * np.where(small, e[:, :-1] * (1 + d / 2), (e[:, 1:] - e[:, :-1]) / np.where(small, 1.0, d))
        cum = np.concatenate([np.zeros((len(mass), 1)), np.cumsum(cell, axis=1)], axis=1)
        ln_norm = np.log(cum[:, -1]) + shift
        return valid, a, cum, ln_norm

    def _warn_off_grid(self, valid):
        if not self._warned_off_grid and not np.all(valid):
            logger.warning(
                f"{self.name}: conditioning mass outside the TOV grid "
                f"[{self._grid.mass_edges[0]}, {self._grid.mass_edges[-1]}]; returning nan. "
                "Restrict the mass prior to the grid or add a Constraint."
            )
            self._warned_off_grid = True

    @xp_wrap
    def rescale(self, val, *, xp=None, **required_variables):
        """Inverse CDF of p(Lambda | mass), exact for the interpolant.

        Within the cell holding the target probability, the remaining mass
        rho satisfies h (e^{a_j + d t} - e^{a_j}) / d = rho, solved for t in
        closed form.
        """
        mass = self._mass(required_variables)
        val, mass = np.broadcast_arrays(np.asarray(val, dtype=float), mass)
        shape = val.shape
        u, mass = val.ravel(), mass.ravel()
        valid, a, cum, _ = self._conditional_tables(mass)
        self._warn_off_grid(valid)

        n_cells = a.shape[1] - 1
        rows = np.arange(len(u))
        target = u * cum[:, -1]
        j = np.clip(np.sum(cum[:, 1:-1] <= target[:, None], axis=1), 0, n_cells - 1)
        h = np.diff(self._grid.log10_lambda_edges)[j]
        rho = target - cum[rows, j]
        a_j = a[rows, j]
        d = a[rows, j + 1] - a_j
        e_j = np.exp(a_j)
        small = np.abs(d) < 1e-8
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(
                small,
                rho / (h * e_j),
                (np.log(np.clip(e_j + rho * d / h, 1e-300, None)) - a_j) / np.where(small, 1.0, d),
            )
        t = np.clip(np.nan_to_num(t, nan=0.0), 0.0, 1.0)
        log10_lambda = self._grid.log10_lambda_edges[j] + t * h
        lam = np.where(valid, 10 ** log10_lambda, np.nan)
        return xp.asarray(lam.reshape(shape))[()]

    @xp_wrap
    def ln_prob(self, val, *, xp=None, **required_variables):
        """ln p(Lambda | mass) = ln p_TOV(mass, Lambda) - ln p_TOV(mass)."""
        mass = self._mass(required_variables)
        val, mass = np.broadcast_arrays(np.asarray(val, dtype=float), mass)
        shape = val.shape
        lam, mass = val.ravel(), mass.ravel()
        valid, _, _, ln_norm = self._conditional_tables(mass)
        with np.errstate(divide="ignore", invalid="ignore"):
            ln_joint = self._grid.ln_pdf_mass_lambda(np.where(valid, mass, self._grid.mass_edges[0]), lam)
        out = np.where(valid & (lam > 0), ln_joint - ln_norm, -np.inf)
        return xp.asarray(out.reshape(shape))[()]

    @xp_wrap
    def prob(self, val, *, xp=None, **required_variables):
        return xp.exp(self.ln_prob(val, xp=xp, **required_variables))

    @xp_wrap
    def cdf(self, val, *, xp=None, **required_variables):
        """CDF of p(Lambda | mass), exact for the interpolant."""
        mass = self._mass(required_variables)
        val, mass = np.broadcast_arrays(np.asarray(val, dtype=float), mass)
        shape = val.shape
        lam, mass = val.ravel(), mass.ravel()
        valid, a, cum, _ = self._conditional_tables(mass)
        edges = self._grid.log10_lambda_edges
        with np.errstate(divide="ignore", invalid="ignore"):
            ell = np.log10(np.clip(lam, 1e-300, None))
        j = np.clip(np.searchsorted(edges, ell) - 1, 0, len(edges) - 2)
        rows = np.arange(len(lam))
        h = np.diff(edges)[j]
        t = np.clip((ell - edges[j]) / h, 0.0, 1.0)
        a_j = a[rows, j]
        d = a[rows, j + 1] - a_j
        small = np.abs(d) < 1e-8
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            partial = h * np.where(small, np.exp(a_j) * t,
                                   (np.exp(a_j + d * t) - np.exp(a_j)) / np.where(small, 1.0, d))
        out = np.clip((cum[rows, j] + partial) / cum[:, -1], 0.0, 1.0)
        out = np.where(valid, out, np.nan)
        return xp.asarray(out.reshape(shape))[()]


def _lambda_tilde_coefficients(eta):
    """The four eta-dependent coefficients of the Wade et al. tidal map.

    See :code:`bilby.gw.conversion.lambda_1_lambda_2_to_lambda_tilde` and
    :code:`..._to_delta_lambda_tilde`, whose definitions these mirror:

        lambda_tilde       = 8 / 13 * (c1 * Lambda_+ + c2 * Lambda_-)
        delta_lambda_tilde = 1 / 2  * (c3 * Lambda_+ + c4 * Lambda_-)

    with Lambda_+/- = Lambda_1 +/- Lambda_2.
    """
    root = np.sqrt(np.clip(1 - 4 * eta, 0, None))
    c1 = 1 + 7 * eta - 31 * eta ** 2
    c2 = root * (1 + 9 * eta - 11 * eta ** 2)
    c3 = root * (1 - 13272 / 1319 * eta + 8944 / 1319 * eta ** 2)
    c4 = (1 - 15910 / 1319 * eta + 32850 / 1319 * eta ** 2 +
          3380 / 1319 * eta ** 3)
    return c1, c2, c3, c4


def _ln_abs_det_lambda_tilde_jacobian(eta):
    """ln |d(lambda_tilde, delta_lambda_tilde) / d(lambda_1, lambda_2)|.

    The Wade et al. map is *linear* in (lambda_1, lambda_2) at fixed masses,

        (lambda_tilde, delta_lambda_tilde)^T = A(eta) (lambda_1, lambda_2)^T,

    so its Jacobian determinant depends on eta alone:

        det A = 8 / 13 * (c2 * c3 - c1 * c4).

    """
    c1, c2, c3, c4 = _lambda_tilde_coefficients(eta)
    det = 8 / 13 * (c2 * c3 - c1 * c4)
    return np.log(np.abs(det))


class TOVMassLambdaTildeJointDist(BaseJointPriorDist):
    """
    Joint distribution of (chirp mass, mass ratio, lambda_tilde,
    delta_lambda_tilde) induced by the TOV mass-Lambda KDE.

    Same input grid as :code:`bilby.gw.prior.TOVJointDist`: a 2D KDE of
    p(mass, log10(Lambda)) built offline from TOV sequences over a population
    of equations of state. Where :code:`TOVJointDist` puts that correlation on
    a single (mass, Lambda) pair, this class propagates it into the sampling
    parameters actually used for BNS runs, so a run in
    (chirp_mass, mass_ratio, lambda_tilde, delta_lambda_tilde) carries the EOS
    correlation rather than an independent uniform tidal prior.

    The underlying model is

        (mass_1, Lambda_1), (mass_2, Lambda_2) ~ p_TOV, independently,
        conditioned on mass_1 >= mass_2,

    pushed through the deterministic maps
    (mass_1, mass_2) -> (chirp_mass, mass_ratio) and, at fixed masses, the
    linear Wade et al. map (Lambda_1, Lambda_2) -> (lambda_tilde,
    delta_lambda_tilde). Both maps are invertible, so the density is exact:

        p(mc, q, lt, dlt) = 2 p(m1, L1) p(m2, L2) * m1^2 / mc / |det A(eta)|

    The full Jacobian is block triangular -- the masses do not depend on the
    tidal parameters -- so it factorises into the usual m1^2 / mc mass term and
    the tidal term 1 / |det A|.

    Note the tidal parameters here are the *dependent* pair: a
    (lambda_tilde, delta_lambda_tilde) rectangle contains points whose implied
    (Lambda_1, Lambda_2) are negative or off the EOS grid. Those get -inf,
    which is correct but means the prior volume is not a box; nested samplers
    should use :meth:`rescale` (via :code:`TOVMassLambdaTildeJointPrior`)
    rather than rejection.
    """

    def __init__(self, grid_file, names=None, bounds=None):
        """
        Parameters
        ==========
        grid_file: str
            Path to an .npz file with arrays ``mass_edges`` (N,),
            ``log10_lambda_edges`` (M,), and ``log_pdf`` (N, M) -- the natural
            log of p(mass, log10(Lambda)) on that grid. The same file used by
            :code:`TOVJointDist`.
        names: list, optional
            Exactly four parameter names, in the order
            [chirp mass, mass ratio, lambda tilde, delta lambda tilde].
            Defaults to
            ``["chirp_mass", "mass_ratio", "lambda_tilde", "delta_lambda_tilde"]``.
        bounds: list, optional
            Bounds for each of the four names. Defaults to the extent implied
            by the grid (see :meth:`default_bounds`); these are outer bounds on
            a non-rectangular support, so being inside them does not imply
            finite probability.
        """
        if names is None:
            names = [
                "chirp_mass", "mass_ratio", "lambda_tilde", "delta_lambda_tilde",
            ]
        if len(names) != 4:
            raise ValueError(
                "TOVMassLambdaTildeJointDist needs exactly four parameter "
                "names: [chirp_mass, mass_ratio, lambda_tilde, "
                "delta_lambda_tilde]"
            )

        self.grid_file = grid_file
        # The single-star grid machinery (marginal mass CDF, conditional
        # Lambda | mass CDF, the KDE interpolant) is reused wholesale; the
        # names given here are internal and never surface to the sampler.
        self._grid = TOVJointDist(grid_file, names=["mass", "lambda"])

        if bounds is None:
            bounds = self.default_bounds()
        super(TOVMassLambdaTildeJointDist, self).__init__(names=names, bounds=bounds)
        # See the note in TOVJointDist: ConditionalPriorDict.rescale groups
        # sibling JointPriors by this string, so it must be unique per instance.
        self.distname = "tov_mc_q_lambda_tilde_" + "_".join(names)

    @property
    def mass_edges(self):
        return self._grid.mass_edges

    @property
    def log10_lambda_edges(self):
        return self._grid.log10_lambda_edges

    def default_bounds(self):
        """Outer bounds on (chirp_mass, mass_ratio, lambda_tilde, delta_lambda_tilde).

        The mass bounds follow analytically from the grid's mass range. The
        tidal map is linear in (Lambda_1, Lambda_2) at fixed eta, so its
        extrema over the grid's Lambda range sit at the corners; those are
        scanned over a mass grid. Deliberately loose -- these only exist to
        reject clearly invalid samples, with :meth:`_ln_prob` doing the real
        support check.
        """
        m_min = float(self.mass_edges[0])
        m_max = float(self.mass_edges[-1])
        lam_min = float(10 ** self.log10_lambda_edges[0])
        lam_max = float(10 ** self.log10_lambda_edges[-1])

        mc_bounds = (
            component_masses_to_chirp_mass(m_min, m_min),
            component_masses_to_chirp_mass(m_max, m_max),
        )
        q_bounds = (m_min / m_max, 1.0)

        masses = np.linspace(m_min, m_max, 200)
        m1, m2 = np.meshgrid(masses, masses, indexing="ij")
        ordered = m1 >= m2
        m1, m2 = m1[ordered], m2[ordered]
        lt_vals, dlt_vals = [], []
        for lam_1 in (lam_min, lam_max):
            for lam_2 in (lam_min, lam_max):
                lt_vals.append(
                    lambda_1_lambda_2_to_lambda_tilde(lam_1, lam_2, m1, m2))
                dlt_vals.append(
                    lambda_1_lambda_2_to_delta_lambda_tilde(lam_1, lam_2, m1, m2))
        lt_vals = np.concatenate(lt_vals)
        dlt_vals = np.concatenate(dlt_vals)

        return [
            (float(mc_bounds[0]), float(mc_bounds[1])),
            (float(q_bounds[0]), float(q_bounds[1])),
            (float(np.min(lt_vals)), float(np.max(lt_vals))),
            (float(np.min(dlt_vals)), float(np.max(dlt_vals))),
        ]

    def _component_rescale(self, samp):
        """Unit cube -> (mass_1, mass_2, lambda_1, lambda_2).

        The four unit-cube coordinates are consumed in order as mass_1,
        then mass_2 given mass_1, then Lambda_1 given mass_1, then Lambda_2
        given mass_2. Which cube coordinate feeds which physical parameter is
        arbitrary: ``rescale`` maps the whole vector at once, so the sampler
        never sees a per-coordinate correspondence.

        The ordering mass_1 >= mass_2 is imposed by the transform itself
        rather than by rejection, keeping the map a smooth bijection:
        mass_1 is drawn from the distribution of the *larger* of two draws,
        whose CDF is F(m)^2, and mass_2 from p(m) truncated to [m_min, m_1].
        """
        mass_1 = self._grid.mass_from_cdf(np.sqrt(samp[:, 0]))
        mass_2 = self._grid.mass_from_cdf(samp[:, 1] * self._grid.mass_cdf(mass_1))
        lambda_1 = 10 ** self._grid.log10_lambda_from_cdf(samp[:, 2], mass_1)
        lambda_2 = 10 ** self._grid.log10_lambda_from_cdf(samp[:, 3], mass_2)
        return mass_1, mass_2, lambda_1, lambda_2

    def _rescale(self, samp, **kwargs):
        samp = np.asarray(samp)
        mass_1, mass_2, lambda_1, lambda_2 = self._component_rescale(samp)
        return np.column_stack([
            component_masses_to_chirp_mass(mass_1, mass_2),
            mass_2 / mass_1,
            lambda_1_lambda_2_to_lambda_tilde(lambda_1, lambda_2, mass_1, mass_2),
            lambda_1_lambda_2_to_delta_lambda_tilde(lambda_1, lambda_2, mass_1, mass_2),
        ])

    def _sample(self, size, *, random_state=None, **kwargs):
        # As in TOVJointDist: honour random_state so seeded runs reproduce and
        # pool workers get distinct streams.
        rng = random.resolve_random_state(random_state)
        return self._rescale(rng.uniform(0, 1, size=(size, 4)))

    @xp_wrap
    def _ln_prob(self, samp, lnprob, outbounds, *, xp=None):
        # As in TOVJointDist, the KDE is evaluated with a scipy
        # RegularGridInterpolator, so the work is done in numpy and converted
        # back on the way out; an xp of cupy/jax would fail here.
        samp = np.asarray(samp)
        chirp_mass, mass_ratio = samp[:, 0], samp[:, 1]
        lambda_tilde, delta_lambda_tilde = samp[:, 2], samp[:, 3]

        with np.errstate(invalid="ignore", divide="ignore"):
            total_mass = chirp_mass_and_mass_ratio_to_total_mass(
                chirp_mass, mass_ratio)
            mass_1, mass_2 = total_mass_and_mass_ratio_to_component_masses(
                mass_ratio, total_mass)
            lambda_1, lambda_2 = lambda_tilde_delta_lambda_tilde_to_lambda_1_lambda_2(
                lambda_tilde, delta_lambda_tilde, mass_1, mass_2)

            # Points of the (lambda_tilde, delta_lambda_tilde) plane that imply
            # a negative Lambda for either star are unphysical, not merely
            # unlikely; likewise mass_ratio outside (0, 1]. Substituting a
            # harmless in-grid value at those points keeps nan out of the KDE
            # interpolant; they are masked to -inf below.
            valid = (
                (mass_ratio > 0) & (mass_ratio <= 1) & (chirp_mass > 0) &
                (lambda_1 > 0) & (lambda_2 > 0) &
                np.isfinite(mass_1) & np.isfinite(mass_2) &
                np.isfinite(lambda_1) & np.isfinite(lambda_2)
            )
            fill_mass = float(self.mass_edges[0])
            fill_lambda = float(10 ** self.log10_lambda_edges[0])
            mass_1 = np.where(valid, mass_1, fill_mass)
            mass_2 = np.where(valid, mass_2, fill_mass)
            lambda_1 = np.where(valid, lambda_1, fill_lambda)
            lambda_2 = np.where(valid, lambda_2, fill_lambda)
            eta = mass_1 * mass_2 / (mass_1 + mass_2) ** 2

            lnprob_vals = (
                np.log(2.0)  # ordering mass_1 >= mass_2 doubles the density
                + self._grid.ln_pdf_mass_lambda(mass_1, lambda_1)
                + self._grid.ln_pdf_mass_lambda(mass_2, lambda_2)
                # d(mass_1, mass_2) / d(chirp_mass, mass_ratio) = mass_1^2 / chirp_mass
                + 2 * np.log(mass_1) - np.log(np.where(valid, chirp_mass, 1.0))
                # d(lambda_1, lambda_2) / d(lambda_tilde, delta_lambda_tilde)
                - _ln_abs_det_lambda_tilde_jacobian(eta)
            )
        lnprob_vals = np.where(valid, lnprob_vals, -np.inf)
        return xp.where(xp.asarray(outbounds), -np.inf, xp.asarray(lnprob_vals))

    @xp_wrap
    def ln_prob(self, value, *, xp=None):
        """Log-probability of a sample, as a scalar for single samples.

        See :meth:`TOVJointDist.ln_prob` for why this override exists.
        """
        lnprob = super().ln_prob(value, xp=xp)
        if getattr(lnprob, "shape", None) == (1,):
            lnprob = lnprob[0]
        return lnprob


class TOVMassLambdaTildeJointPrior(JointPrior):
    """
    A prior distribution following the TOV-informed joint distribution of
    (chirp_mass, mass_ratio, lambda_tilde, delta_lambda_tilde), for one of
    those four parameters.

    One instance is created per parameter, all sharing a single
    :code:`bilby.gw.prior.TOVMassLambdaTildeJointDist`, e.g.

    .. code-block:: python

        dist = TOVMassLambdaTildeJointDist("tov_mass_lambda_joint_kde_grid.npz")
        priors["chirp_mass"] = TOVMassLambdaTildeJointPrior(dist, "chirp_mass")
        priors["mass_ratio"] = TOVMassLambdaTildeJointPrior(dist, "mass_ratio")
        priors["lambda_tilde"] = TOVMassLambdaTildeJointPrior(dist, "lambda_tilde")
        priors["delta_lambda_tilde"] = TOVMassLambdaTildeJointPrior(
            dist, "delta_lambda_tilde")
    """

    def __init__(self, dist, name=None, latex_label=None, unit=None):
        """
        Parameters
        ----------
        dist: bilby.gw.prior.TOVMassLambdaTildeJointDist
            The base joint probability.
        name: str
            The name of the parameter; one of dist.names.
        latex_label: str
            Latex label used for plotting.
        unit: str
            The unit of the parameter.
        """
        if not isinstance(dist, TOVMassLambdaTildeJointDist):
            raise JointPriorDistError(
                "dist object must be instance of TOVMassLambdaTildeJointDist")
        super(TOVMassLambdaTildeJointPrior, self).__init__(
            dist=dist, name=name, latex_label=latex_label, unit=unit)

    def sample(self, size=1, *, random_state=None, **kwargs):
        """Draw a sample, returning a scalar rather than a 0-d array.

        See :meth:`TOVJointPrior.sample` for why this override exists.
        """
        return super().sample(size=size, random_state=random_state, **kwargs)[()]
