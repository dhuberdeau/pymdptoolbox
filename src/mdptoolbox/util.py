# -*- coding: utf-8 -*-
"""Markov Decision Process (MDP) Toolbox: ``util`` module
======================================================

The ``util`` module provides functions to check that an MDP is validly
described. There are also functions for working with MDPs while they are being
solved, and for manipulating raw data into formats that the MDP functions can
solve.

Available functions
-------------------

:func:`~mdptoolbox.util.check`
    Check that an MDP is properly defined
:func:`~mdptoolbox.util.checkSquareStochastic`
    Check that a matrix is square and stochastic
:func:`~mdptoolbox.util.getSpan`
    Calculate the span of an array
:func:`~mdptoolbox.util.isNonNegative`
    Check if a matrix has only non-negative elements
:func:`~mdptoolbox.util.isSquare`
    Check if a matrix is square
:func:`~mdptoolbox.util.isStochastic`
    Check if a matrix is row stochastic
:func:`~mdptoolbox.util.fit`
    Compute transition and reward matricies from a set of trajectories
:func:`~mdptoolbox.util.predict`
    Predict the action sequences from a set of trajectories
:func:`~mdptoolbox.util.score`
    Score the deviation of a trajectory from optimal policy
:func:`~mdptoolbox.util.simulate`
    Simulate a trajectory through state space under the current MDP policy

"""

# Copyright (c) 2011-2015 Steven A. W. Cordwell
# Copyright (c) 2009 INRA
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#   * Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#   * Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#   * Neither the name of the <ORGANIZATION> nor the names of its contributors
#     may be used to endorse or promote products derived from this software
#     without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import numpy as _np

import mdptoolbox.error as _error

import mdptoolbox as _mdp

_MDPERR = {
"mat_nonneg" :
    "Transition probabilities must be non-negative.",
"mat_square" :
    "A transition probability matrix must be square, with dimensions S×S.",
"mat_stoch" :
    "Each row of a transition probability matrix must sum to one (1).",
"obj_shape" :
    "Object arrays for transition probabilities and rewards "
    "must have only 1 dimension: the number of actions A. Each element of "
    "the object array contains an SxS ndarray or matrix.",
"obj_square" :
    "Each element of an object array for transition "
    "probabilities and rewards must contain an SxS ndarray or matrix; i.e. "
    "P[a].shape = (S, S) or R[a].shape = (S, S).",
"P_type" :
    "The transition probabilities must be in a numpy array; "
    "i.e. type(P) is ndarray.",
"P_shape" :
    "The transition probability array must have the shape "
    "(A, S, S)  with S : number of states greater than 0 and A : number of "
    "actions greater than 0. i.e. R.shape = (A, S, S)",
"PR_incompat" :
    "Incompatibility between P and R dimensions.",
"R_type" :
    "The rewards must be in a numpy array; i.e. type(R) is "
    "ndarray, or numpy matrix; i.e. type(R) is matrix.",
"R_shape" :
    "The reward matrix R must be an array of shape (A, S, S) or "
    "(S, A) with S : number of states greater than 0 and A : number of "
    "actions greater than 0. i.e. R.shape = (S, A) or (A, S, S)."
}


def _checkDimensionsListLike(arrays):
    """Check that each array in a list of arrays has the same size.

    """
    dim1 = len(arrays)
    dim2, dim3 = arrays[0].shape
    for aa in range(1, dim1):
        dim2_aa, dim3_aa = arrays[aa].shape
        if (dim2_aa != dim2) or (dim3_aa != dim3):
            raise _error.InvalidError(_MDPERR["obj_square"])
    return dim1, dim2, dim3


def _checkRewardsListLike(reward, n_actions, n_states):
    """Check that a list-like reward input is valid.

    """
    try:
        lenR = len(reward)
        if lenR == n_actions:
            dim1, dim2, dim3 = _checkDimensionsListLike(reward)
        elif lenR == n_states:
            dim1 = n_actions
            dim2 = dim3 = lenR
        else:
            raise _error.InvalidError(_MDPERR["R_shape"])
    except AttributeError:
        raise _error.InvalidError(_MDPERR["R_shape"])
    return dim1, dim2, dim3


def isSquare(matrix):
    """Check that ``matrix`` is square.

    Returns
    =======
    is_square : bool
        ``True`` if ``matrix`` is square, ``False`` otherwise.

    """
    try:
        try:
            dim1, dim2 = matrix.shape
        except AttributeError:
            dim1, dim2 = _np.array(matrix).shape
    except ValueError:
        return False
    if dim1 == dim2:
        return True
    return False


def isStochastic(matrix):
    """Check that ``matrix`` is row stochastic.

    Returns
    =======
    is_stochastic : bool
        ``True`` if ``matrix`` is row stochastic, ``False`` otherwise.

    """
    try:
        absdiff = (_np.abs(matrix.sum(axis=1) - _np.ones(matrix.shape[0])))
    except AttributeError:
        matrix = _np.array(matrix)
        absdiff = (_np.abs(matrix.sum(axis=1) - _np.ones(matrix.shape[0])))
    return (absdiff.max() <= 10*_np.spacing(_np.float64(1)))


def isNonNegative(matrix):
    """Check that ``matrix`` is row non-negative.

    Returns
    =======
    is_stochastic : bool
        ``True`` if ``matrix`` is non-negative, ``False`` otherwise.

    """
    try:
        if (matrix >= 0).all():
            return True
    except (NotImplementedError, AttributeError, TypeError):
        try:
            if (matrix.data >= 0).all():
                return True
        except AttributeError:
            matrix = _np.array(matrix)
            if (matrix.data >= 0).all():
                return True
    return False


def checkSquareStochastic(matrix):
    """Check if ``matrix`` is a square and row-stochastic.

    To pass the check the following conditions must be met:

    * The matrix should be square, so the number of columns equals the
      number of rows.
    * The matrix should be row-stochastic so the rows should sum to one.
    * Each value in the matrix must be positive.

    If the check does not pass then a mdptoolbox.util.Invalid

    Arguments
    ---------
    ``matrix`` : numpy.ndarray, scipy.sparse.*_matrix
        A two dimensional array (matrix).

    Notes
    -----
    Returns None if no error has been detected, else it raises an error.

    """
    if not isSquare(matrix):
        raise _error.SquareError
    if not isStochastic(matrix):
        raise _error.StochasticError
    if not isNonNegative(matrix):
        raise _error.NonNegativeError


def check(P, R):
    """Check if ``P`` and ``R`` define a valid Markov Decision Process (MDP).

    Let ``S`` = number of states, ``A`` = number of actions.

    Arguments
    ---------
    P : array
        The transition matrices. It can be a three dimensional array with
        a shape of (A, S, S). It can also be a one dimensional arraye with
        a shape of (A, ), where each element contains a matrix of shape (S, S)
        which can possibly be sparse.
    R : array
        The reward matrix. It can be a three dimensional array with a
        shape of (S, A, A). It can also be a one dimensional array with a
        shape of (A, ), where each element contains matrix with a shape of
        (S, S) which can possibly be sparse. It can also be an array with
        a shape of (S, A) which can possibly be sparse.

    Notes
    -----
    Raises an error if ``P`` and ``R`` do not define a MDP.

    Examples
    --------
    >>> import mdptoolbox, mdptoolbox.example
    >>> P_valid, R_valid = mdptoolbox.example.rand(100, 5)
    >>> mdptoolbox.util.check(P_valid, R_valid) # Nothing should happen
    >>>
    >>> import numpy as np
    >>> P_invalid = np.random.rand(5, 100, 100)
    >>> mdptoolbox.util.check(P_invalid, R_valid) # Raises an exception
    Traceback (most recent call last):
    ...
    StochasticError:...

    """
    # Checking P
    try:
        if P.ndim == 3:
            aP, sP0, sP1 = P.shape
        elif P.ndim == 1:
            aP, sP0, sP1 = _checkDimensionsListLike(P)
        else:
            raise _error.InvalidError(_MDPERR["P_shape"])
    except AttributeError:
        try:
            aP, sP0, sP1 = _checkDimensionsListLike(P)
        except AttributeError:
            raise _error.InvalidError(_MDPERR["P_shape"])
    msg = ""
    if aP <= 0:
        msg = "The number of actions in P must be greater than 0."
    elif sP0 <= 0:
        msg = "The number of states in P must be greater than 0."
    if msg:
        raise _error.InvalidError(msg)
    # Checking R
    try:
        ndimR = R.ndim
        if ndimR == 1:
            aR, sR0, sR1 = _checkRewardsListLike(R, aP, sP0)
        elif ndimR == 2:
            sR0, aR = R.shape
            sR1 = sR0
        elif ndimR == 3:
            aR, sR0, sR1 = R.shape
        else:
            raise _error.InvalidError(_MDPERR["R_shape"])
    except AttributeError:
        aR, sR0, sR1 = _checkRewardsListLike(R, aP, sP0)
    msg = ""
    if sR0 <= 0:
        msg = "The number of states in R must be greater than 0."
    elif aR <= 0:
        msg = "The number of actions in R must be greater than 0."
    elif sR0 != sR1:
        msg = "The matrix R must be square with respect to states."
    elif sP0 != sR0:
        msg = "The number of states must agree in P and R."
    elif aP != aR:
        msg = "The number of actions must agree in P and R."
    if msg:
        raise _error.InvalidError(msg)
    # Check that the P's are square, stochastic and non-negative
    for aa in range(aP):
        checkSquareStochastic(P[aa])


def getSpan(array):
    """Return the span of `array`

    span(array) = max array(s) - min array(s)

    """
    return array.max() - array.min()

class MDP_handle(object):

    def __init__(self, S=None, P=None, A=None, R=None, gamma = None):
        self.S = S
        self.P = P
        self.A = A
        self.R = R
        self.gamma = gamma
        self._P_iter = []
        self._S_index = []
        self._solver = None

    def fit(self, X, y, solver="PolicyIteration"):
        """Compute an optimal policy by first estimating a state
        transition matrix and reward matrix from sample trajectories
        through state space specified as input.

        Let S be the number of states, A the number of possible
        actions, N the number of trials, and T the duration of a
        given trajectory.

        Arguments
        ---------
        X : matrix
            A matrix of trajectories through state space. X should have shape
            (N x T).
        y : array
            The actions taken for each state trajectory in X. y should have
            shape (N x 1).
        solver: string
            The preferred MDP solver (default Policy Iteration)

        """
        self._computeTransitionMatrix(X, y)
        self._computeRewardMatrix()

        if self.gamma is None:
            self.gamma = .9

        # Transpose P: MDP solvers need P to be (A,S,S)
        # whereas this module stores P as (S,S,A)
        # note: P must have rows that sum to 1
        P_trans = _np.transpose(self.P, (2,0,1))
        self._solver = _mdp.mdp.PolicyIteration(P_trans, self.R, self.gamma)
        self._solver.run()

        return self

    def predict(self,X):
        # get the optimal policy:
        pi = self._solver.policy

        aq_pred = _np.empty((_np.shape(X)[0], _np.shape(X)[-1]))
        for i_trial in range(0,len(X)):
            sq = self._stateTransitionSequence(X[i_trial])
            this_aq = []
            for i_sample in range(0, _np.shape(X)[-1]):
                this_state_ind = self._convertDimensionIndexToStateIndex(
                    sq[:, i_sample])
                aq_pred[i_trial, i_sample] = pi[this_state_ind]
                # this_aq = _np.append(this_aq, pi[this_state_ind])

        # y_pred = self.A(aq_pred)
        y_pred = _np.empty((_np.shape(X)[0], _np.shape(X)[-1]))
        for i_trial in range(0,len(X)):
            y_pred[i_trial, :] = [self.A[
                int(aq_pred[i_trial,i_sample])] for i_sample in range(
                0, _np.shape(X)[-1])]

        return y_pred


    def score(self,X,y):

        y_pred = self.predict(X)
        action_error = _np.subtract(y_pred, y)

        return action_error


    def simulate(self, X0, n_steps):
        # given an initial state, predict a
        # state-transition sequence and action sequence
        # assuming the policy is followed.

        pi = self._solver.policy

        # X_pred = _np.empty((1, _np.shape(X0)[1], n_steps))
        X_pred = []
        y_pred = _np.empty(n_steps)
        sq_ind_pred = _np.empty(n_steps)
        aq_pred = _np.empty(n_steps)

        sq0 = self._stateTransitionSequence(X0[0])
        this_state_ind = self._convertDimensionIndexToStateIndex(sq0)
        for i_sample in range(0, n_steps):
            this_action_ind = pi[this_state_ind]
            this_state_ind = _np.random.choice(
                _np.array(range(0,_np.shape(self.P)[0])),
                p=self.P[this_state_ind, :, this_action_ind])

            aq_pred[i_sample] = this_action_ind
            y_pred[i_sample] = self.A[this_action_ind]

            sq_ind_pred[i_sample] = this_state_ind
            # X_pred[0,:,i_sample] = self.S[
            #     self._convertStateIndexToDimensionIndex(this_state_ind)]

            # X_pred[i_sample] = self._convertStateIndexToDimensionIndex(
                # this_state_ind)
            X_pred = _np.append(X_pred,
                self._convertStateIndexToDimensionIndex(
                this_state_ind))

        X_pred = _np.reshape(X_pred, (1, _np.shape(X0)[1], n_steps))

        return (X_pred, y_pred)

    def setP(self, P):
        self.P = P


    def setS(self, S):
        self.S = S


    def setA(self, A):
        self.A = A


    def _stateTransitionSequence(self, x):
        """Transform a trajectory, x, through a state-space, S, into a
        sequence of states and actions.

        The object must have the following already defined:
        S : array
            An array of arrays defining the discritization of the state-
            space. There should be one array for each dimension, and the
            values are the state borders along that dimension. The
            output of this function will be denominated as the index
            of each of these arrays.

        Arguments
        ---------
        x : array
            A trajectory through state-space. One array for each
            dimension of the state-space. Each array should have equal
            numbers of elements.

        Returns
        =======
        sq : array
            The sequence of states, discritized according to S.


        """
        sq = _np.empty((_np.shape(self.S)[0], _np.shape(x)[-1]))

        for i_dim in range(len(x)):
            sq[i_dim, :] = _np.digitize(x[i_dim], self.S[i_dim])

        return sq

    def _actionTransitionSequence(self, a):
        """Transform a trajectory, x, through a state-space, S, into a
        sequence of states and actions.

        The object must have the following already defined:
        A : array
            An array defining the discritization of the action-
            space. There should be one array, and the
            values are the bins into which to discritize the actions, a.

        Arguments
        ---------
        a : array
            The actions taken for each state value in x. a should have
            the same number of elements as each array in x.

        Returns
        =======
        aq : array
            The sequence of actions, discritized according to A.


        """

        aq = _np.digitize(a, self.A)

        return aq


    def _incrementTransitionMatrix(self, sq, aq):
        """Add a trajectory defined as a state sequence, sq, to a
        growing state transition matrix, P, given action sequence aq.

        The object must already have the following defined:
        S : array
            A state space definition
        A : array
            An action space definition

        Arguments
        ---------
        sq : array
            A sequence of state transitions, denominated as indicies
            of the state space defined elsewhere in S.
        aq : array
            A sequence of actions, denominated as indicies of the action space
            defined elsewhere in A.

        """

        if len(self._P_iter) == 0:
            # count up the number of states in S
            dims = []
            for i_dim in range(0, len(self.S)):
                dims = _np.append(dims, len(self.S[i_dim]))
            n_states = int(_np.prod(dims))

            # define the iterator for the transition
            # matrix
            self._P_iter = _np.zeros((n_states,n_states,len(self.A)))

            # define the overall state index from
            # state definition S
            state_range = _np.array(range(0, n_states))
            dims_int = [int(dims[i_]) for i_ in range(len(dims))]
            self._S_index = _np.reshape(state_range, dims_int)

        # increment _P_iter
        for i_sample in range(0,len(sq)-1):
            this_state_ind = self._convertDimensionIndexToStateIndex(
                    sq[:,i_sample])
            # this_state_ind = self._S_index[dim_state_inds]

            next_state_ind = self._convertDimensionIndexToStateIndex(
                    sq[:,i_sample + 1])
            # next_state_ind = self._S_index[dim_state_inds]

            # dim_state_inds = _np.array([], int)
            # for i_dim in range(0,len(sq[i_sample+1])):
            #     dim_state_inds = np.append(dim_state_inds,
            #         int(sq[i_sample+1][i_dim]))
            # next_state_ind = self._S_index[tuple(dim_state_inds)]

            this_action_ind = int(aq[i_sample])

            self._P_iter[this_state_ind][
                    next_state_ind][
                    this_action_ind] = self._P_iter[
                        this_state_ind][
                        next_state_ind][
                        this_action_ind] + 1

        return self


    def _computeTransitionMatrix(self, X, a):

        for i_trial in range(0, len(X)):
            sq = self._stateTransitionSequence(
                X[i_trial])
            aq = self._actionTransitionSequence(
                 a[i_trial])
            self._incrementTransitionMatrix(sq, aq)

        P_0 = _np.copy(self._P_iter)
        for i_action in range(0, _np.shape(self._P_iter)[2]):
            P_action = P_0[:,:,i_action]
            P_row_sum = _np.sum(P_action, 1)
            P_prior = [1/_np.shape(P_0)[0] for i_ in
                         range(0, _np.shape(P_0)[0])]
            for i_state in range(_np.shape(P_action)[0]):
                # P_action[i_state, :] = _np.divide(
                #     P_action[i_state, :], P_row_sum[i_state])
                if P_row_sum[i_state] == 0:
                    # no samples in this row
                    P_action[i_state, :] = P_prior
                else:
                    likelihood = _np.divide(P_action[i_state, :],
                            P_row_sum[i_state])
                    posterior = [likelihood[i_]*P_prior[i_]
                        for i_ in range(0, len(P_prior))]
                    P_action[i_state, :] = _np.divide(posterior,
                        _np.sum(posterior))
            P_0[:,:,i_action] = P_action

        self.P = P_0
        return self


    def _computeRewardMatrix(self):
        # Assume all actions have equally reward/cose, and define
        # state rewards as the column density across states

        P_0 = _np.copy(self._P_iter)
        R_0 = _np.zeros((_np.shape(self._P_iter)[0],
            _np.shape(self._P_iter)[-1]))
        for i_action in range(0, len(self.A)):
            P_action = P_0[:,:,i_action]
            P_col_sum = _np.sum(P_action, 0)
            P_total_sum = _np.sum(P_col_sum)
            if P_total_sum == 0:
                P_density = _np.zeros(_np.shape(R_0)[0])
            else:
                P_density = _np.divide(P_col_sum, P_total_sum)
            R_0[:, i_action] = P_density

        self.R = R_0

        return self

    def _convertDimensionIndexToStateIndex(self, sq_sample):

        dim_state_inds = _np.empty(_np.shape(sq_sample)[0], dtype=int)
        for i_dim in range(0,_np.shape(sq_sample)[0]):
            dim_state_inds[i_dim] = int(sq_sample[i_dim])

        this_state_ind = self._S_index[tuple(dim_state_inds)]

        return this_state_ind

    def _convertStateIndexToDimensionIndex(self, sq_ind_sample):

        sq_sample = _np.where(self._S_index == sq_ind_sample)
        return sq_sample
