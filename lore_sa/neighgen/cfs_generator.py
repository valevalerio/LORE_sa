from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split

from lore_sa.neighgen.neighgen import NeighborhoodGenerator
import numpy as np

import warnings

warnings.filterwarnings("ignore")

__all__ = ["NeighborhoodGenerator","CFSGenerator"]

class CFSGenerator(NeighborhoodGenerator):
    def __init__(self, bb_predict, feature_values, features_map, nbr_features, nbr_real_features,
                 numeric_columns_index, ocr=0.1, n_search=10000,n_batch=1000,lower_threshold=0,upper_threshold=4,
            kind="gaussian_matched",sampling_kind=None, stopping_ratio=0.01,
            check_upper_threshold=True, final_counterfactual_search=True,
            custom_sampling_threshold=None, custom_closest_counterfactual=None,
            n=500, balance=False, verbose=False, cut_radius=None, forced_balance_ratio=0.5, downward_only=True, encdec=None):
        """

        :param bb_predict:
        :param feature_values:
        :param features_map:
        :param nbr_features:
        :param nbr_real_features:
        :param numeric_columns_index:
        :param ocr:
        :param n_search:
        :param n_batch:
        :param lower_threshold:
        :param upper_threshold:
        :param kind:
        :param sampling_kind:
        :param stopping_ratio:
        :param check_upper_threshold:
        :param final_counterfactual_search:
        :param custom_sampling_threshold:
        :param custom_closest_counterfactual:
        :param n:
        :param balance:
        :param verbose:
        :param cut_radius:
        :param forced_balance_ratio:
        :param downward_only:
        :param encdec:
        """
        super(CFSGenerator, self).__init__(bb_predict=bb_predict, feature_values=feature_values, features_map=features_map,
                                           nbr_features=nbr_features,nbr_real_features=nbr_real_features,
                                           numeric_columns_index=numeric_columns_index, ocr=ocr, encdec=encdec)
        self.n_search = n_search
        self.n_batch = n_batch
        self.lower_threshold = lower_threshold
        self.upper_threshold = upper_threshold
        self.kind = kind
        self.sampling_kind = sampling_kind
        self.verbose = verbose
        self.check_upper_threshold = check_upper_threshold
        self.final_counterfactual_search = final_counterfactual_search
        self.stopping_ratio = stopping_ratio
        self.n = n
        self.forced_balance_ratio = forced_balance_ratio
        self.custom_closest_counterfactual = custom_closest_counterfactual
        self.balance = balance
        self.custom_sampling_threshold = custom_sampling_threshold
        self.cut_radius = cut_radius
        self.closest_counterfactual = None
        self.downward_only = downward_only


    def generate(self, x, num_samples=1000, **kwargs):
        """

        :param x:
        :param num_samples:
        :param kwargs:
        :return:
        """
        x_label = self.apply_bb_predict(x.reshape(1, -1))
        if (self.closest_counterfactual is None) and (self.custom_closest_counterfactual is None):
            self.counterfactual_search(x,x_label=x_label, **kwargs)

        self.kind = self.sampling_kind if self.sampling_kind is not None else self.kind

        Z = self.neighborhood_sampling(x,x_label=x_label,**kwargs)
        #check di Z: se contiene dei nan o degli infiniti li devo togliere
        Z = np.nan_to_num(Z)
        return Z

    def counterfactual_search(self,x,x_label, **kwargs):
        x = x.reshape(1,-1)
        self.closest_counterfactual, self.best_threshold = self.binary_sampling_search(x,x_label, downward_only=self.downward_only, **kwargs)
        return self.closest_counterfactual, self.best_threshold

    def binary_sampling_search(self, x, x_label, downward_only=True, **kwargs):
        if self.verbose:
            print("Binary sampling search:", self.kind)

        # sanity check for the upper threshold
        if self.check_upper_threshold:
            if self.verbose:
                print('---     ', self.n, self.n_batch, int(self.n / self.n_batch))
            print('binary sampling search ', self.n, self.n_batch)
            if int(self.n / float(self.n_batch)) < 1:
                raise Exception('Change the value of n or n_batch')
            for i in range(int(self.n / float(self.n_batch))):
                Z = self.vicinity_sampling(
                    x,
                    n=self.n_batch,
                    threshold=self.upper_threshold,
                    **kwargs
                )

                y = self.apply_bb_predict(Z)
                if not np.all(y == x_label):
                    break
            if i == list(range(int(self.n / self.n_batch)))[-1]:
                raise Exception("No counterfactual found, increase upper threshold or n_search.")

        change_lower = False
        latest_working_threshold = self.upper_threshold
        Z_counterfactuals = list()
        if self.verbose:
            print('lower threshold, upper threshold ', self.lower_threshold, self.upper_threshold)
        while self.lower_threshold / self.upper_threshold < self.stopping_ratio:
            if change_lower:
                if downward_only:
                    break
                self.lower_threshold = threshold
            threshold = (self.lower_threshold + self.upper_threshold) / 2
            change_lower = True
            if self.verbose:
                print("   Testing threshold value:", threshold)
            for i in range(int(self.n / self.n_batch)):
                Z = self.vicinity_sampling(
                    x,
                    n=self.n_batch,
                    threshold=threshold,
                    **kwargs
                )

                y = self.apply_bb_predict(Z)
                if not np.all(y == x_label):  # if we found already some counterfactuals
                    counterfactuals_idxs = np.argwhere(y != x_label).ravel()
                    Z_counterfactuals.append(Z[counterfactuals_idxs])
                    latest_working_threshold = threshold
                    self.upper_threshold = threshold
                    change_lower = False
                    break
        if self.verbose:
            print("   Best threshold found:", latest_working_threshold)
        if self.final_counterfactual_search:
            if self.verbose:
                print("   Final counterfactual search... (this could take a while)", end=" ")
            Z = self.vicinity_sampling(
                x,
                n = self.n,
                threshold=latest_working_threshold,
                **kwargs
            )
            y = self.apply_bb_predict(Z)
            counterfactuals_idxs = np.argwhere(y != x_label).ravel()
            Z_counterfactuals.append(Z[counterfactuals_idxs])
            if self.verbose:
                print("Done!")
        Z_counterfactuals = np.concatenate(Z_counterfactuals)
        closest_counterfactual = min(Z_counterfactuals, key=lambda p: sum((p - x.ravel()) ** 2))
        return closest_counterfactual, latest_working_threshold

    def neighborhood_sampling(self, x, x_label, custom_closest_counterfactual=None, custom_sampling_threshold=None, **kwargs):
        if custom_closest_counterfactual is not None:
            self.closest_counterfactual = custom_closest_counterfactual
        if self.cut_radius:
            self.best_threshold = np.linalg.norm(x - self.closest_counterfactual)
            if self.verbose:
                print("Setting new threshold at radius:", self.best_threshold)
            if self.kind not in ["uniform_sphere"]:
                warnings.warn("cut_radius=True, but for the method " + self.kind + " the threshold is not a radius.")
        if custom_sampling_threshold is not None:
            self.best_threshold = custom_sampling_threshold
            if self.verbose:
                print("Setting custom threshold:", self.best_threshold)
        Z = self.vicinity_sampling(
            self.closest_counterfactual.reshape(1,-1),
            n=self.n,
            threshold=self.best_threshold,
            **kwargs
        )

        if self.forced_balance_ratio is not None:
            y = self.apply_bb_predict(Z)
            y = 1 * (y == x_label)
            n_minority_instances = np.unique(y, return_counts=True)[1].min()
            if (n_minority_instances / self.n) < self.forced_balance_ratio:
                if self.verbose:
                    print("Forced balancing neighborhood...", end=" ")
                n_desired_minority_instances = int(self.forced_balance_ratio * self.n)
                n_desired_majority_instances = self.n - n_desired_minority_instances
                minority_class = np.argmin(np.unique(y, return_counts=True)[1])
                sampling_strategy = n_desired_minority_instances / n_desired_majority_instances
                while n_minority_instances < n_desired_minority_instances:
                    Z_ = self.vicinity_sampling(
                        self.closest_counterfactual.reshape(1,-1),
                        n=self.n_batch,
                        threshold=self.best_threshold if custom_sampling_threshold is None else custom_sampling_threshold,
                        **kwargs
                    )

                    y_ = self.apply_bb_predict(Z_)
                    y_ = 1 * (y_ == x_label)
                    n_minority_instances += np.unique(y_, return_counts=True)[1][minority_class]
                    Z = np.concatenate([Z, Z_])
                    y = np.concatenate([y, y_])
                rus = RandomUnderSampler(random_state=0, sampling_strategy=sampling_strategy)
                Z, y = rus.fit_resample(Z, y)
                if len(Z) > self.n:
                    Z, _ = train_test_split(Z, train_size=self.n, stratify=y)
                if self.verbose:
                    print("Done!")

        if self.balance:
            if self.verbose:
                print("Balancing neighborhood...", end=" ")
            rus = RandomUnderSampler(random_state=0)
            y = self.apply_bb_predict(Z)
            y = 1 * (y == x_label)
            Z, _ = rus.fit_resample(Z, y)
            if self.verbose:
                print("Done!")
        return Z

    def vicinity_sampling(self, x, n, threshold=None,**kwargs):
        if self.verbose:
            print("\nSampling -->", self.kind)
        if self.kind == "gaussian":
            Z = self.gaussian_vicinity_sampling(x, threshold, n)
        elif self.kind == "gaussian_matched":
            Z = self.gaussian_matched_vicinity_sampling(x, threshold, n)
        elif self.kind == "gaussian_global":
            Z = self.gaussian_global_sampling(x, n)
        elif self.kind == "uniform_sphere":
            Z = self.uniform_sphere_vicinity_sampling(x, n, threshold)
        elif self.kind == "uniform_sphere_scaled":
            Z = self.uniform_sphere_scaled_vicinity_sampling(x, n, threshold)
        else:
            raise Exception("Vicinity sampling kind not valid", self.kind)
        return Z

    def gaussian_vicinity_sampling(self, z, epsilon, n=1):
        return z + (np.random.normal(size=(n, z.shape[1])) * epsilon)

    def gaussian_vicinity_sampling(self, z, epsilon, n=1):
        return z + (np.random.normal(size=(n, z.shape[1])) * epsilon)

    def gaussian_global_sampling(self, z, n=1):
        return np.random.normal(size=(n, z.shape[1]))

    def uniform_sphere_origin(self, n, d, r=1):
        """Generate "num_points" random points in "dimension" that have uniform probability over the unit ball scaled
        by "radius" (length of points are in range [0, "radius"]).

        Parameters
        ----------
        n : int
            number of points to generate
        d : int
            dimensionality of each point
        r : float
            radius of the sphere

        Returns
        -------
        array of shape (n, d)
            sampled points
        """
        # First generate random directions by normalizing the length of a
        # vector of random-normal values (these distribute evenly on ball).
        random_directions = np.random.normal(size=(d, n))
        random_directions /= np.linalg.norm(random_directions, axis=0)
        # Second generate a random radius with probability proportional to
        # the surface area of a ball with a given radius.
        random_radii = np.random.random(n) ** (1 / d)
        # Return the list of random (direction & length) points.
        return r * (random_directions * random_radii).T

    def uniform_sphere_vicinity_sampling(self, z, n=1, r=1):
        Z = self.uniform_sphere_origin(n, z.shape[1], r)
        self.translate(Z, z)
        return Z

    def uniform_sphere_scaled_vicinity_sampling(self, z, n=1, threshold=1):
        Z = self.uniform_sphere_origin(n, z.shape[1], r=1)
        Z *= threshold
        self.translate(Z, z)
        return Z

    def translate(self, X, center):
        """Translates a origin centered array to a new center

        Parameters
        ----------
        X : array
            data to translate centered in the axis origin
        center : array
            new center point

        Returns
        -------
        None
        """
        for axis in range(center.shape[-1]):
            X[..., axis] += center[..., axis]