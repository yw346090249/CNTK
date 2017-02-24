from cntk import cntk_py
from cntk.device import DeviceDescriptor
from cntk.utils import typemap, sanitize_var_map, sanitize_batch, variable_value_to_seq

from cntk.utils.swig_helper import map_if_possible
from cntk.ops.variables import Variable
from enum import Enum, unique
import numpy as np


@unique
class CloneMethod(Enum):
    '''
    Describes different ways how :func:`~cntk.ops.functions.Function.clone`
    works.
    '''

    share = 'share'
    '''
    Parameters are shared between the Function being cloned and the new clone
    '''

    clone = 'clone'
    '''
    New learnable parameters are created and initialized with the current values of the
    corresponding parameters of the Function being cloned
    '''

    freeze = 'freeze'
    '''
    Parameters are cloned and made immutable; i.e. Constants in the new clone
    (e.g. for use as a fixed feature extractor)
    '''

class Function(cntk_py.Function):
    '''
    Base class of all primitive tensor operators.

    If it has only one output, one can invoke Variable methods on it, which it
    will relay to its only output.
    '''

    # define input shapes, in-place
    # e.g.
    # model.declare_args(42)
    # pass a list of objects that define the dimensions etc. of the placeholders
    # Currently you can pass either
    def declare_args(self, *arg_types):
        placeholders = self.placeholders  # the unbound parameters to fill in
        if len(arg_types) != len(placeholders):
            raise TypeError("CNTK Function.declare_inputs() expected {} arguments, got {}".format(len(placeholders), len(arg_types)))
        def to_input(arg):
            if isinstance(arg, cntk_py.Variable):
                return arg
            else:
                from cntk import input_variable
                return input_variable(arg)
        args = [to_input(arg) for arg in arg_types]
        self.replace_placeholders(dict(zip(placeholders, args)))


    # call a function, i.e. clone with all placeholders/inputs replaced
    def __call__(self, *args):
        if not isinstance(args, tuple):  # normalize single argument into tuple
            args = (args,)
        # flatten args to a list. Note it may be a a tuple or even a nested tree of tuples, e.g. LSTM (x, (h, c))
        def flatten_tuple(args):
            if not isinstance(args, tuple): # not a tuple: singleton; create a singleton tuple
                return (args,)
            from operator import add
            from functools import reduce
            return reduce(add, [(flatten_tuple(item)) for item in args])
        args = list(flatten_tuple(args))  # normalize nested arg tuples into flat tuple  --TODO: is there a standard function to do this?
        # TODO: This should not be necessary, or go into Function.replace_placeholders()
        def _output_of(arg):  # helper to get the output of an arg; use arg itself if no output() method (that'd be a Variable)
            try:
                return arg.output
            except AttributeError:
                return arg  # Variables have no output()
        args = [_output_of(arg) for arg in args]  # normalize args to their outputs  --BUGBUG: without: "TypeError: cannot convert value of dictionary to CNTK::Variable "
        #from cntk.ops import combine
        #args = [combine([arg]) for arg in args]  # BUGBUG: without: "TypeError: cannot convert value of dictionary to CNTK::Variable "
        placeholders = self.placeholders  # the unbound parameters to fill in
        if len(args) != len(placeholders):
            raise TypeError("CNTK Function expected {} arguments, got {}".format(len(placeholders), len(args)))
        return self.clone(CloneMethod.share, dict(zip(placeholders, args)))

    # forward function composition (other o self)
    def __rshift__(self, other):
        return other(self)

    # backward function composition (self o other)
    def __lshift__(self, other):
        return self(other)

    def __getattr__(self, name):
        # If something is not found in Function, look it up in its output
        # variable, if it has only one.
        if not hasattr(Variable, name) or name.startswith('_') or \
                name in ['outputs', 'output', 'this']:
            # These should not be looked up in self's output.
            # 'outputs' and 'output' are required to fetch the attribute for
            # in the Variable.
            # 'this' is required for Swig and needs to be thrown if the
            # object is created the first time.
            raise AttributeError("neither Function nor its output variable"
                    " has '%s'"%name)

        outputs = self.__getattribute__('outputs')
        if len(outputs) != 1:
            raise AttributeError("Function does not have '%s' and it cannot "
                    "be looked up in its outputs because it does not have "
                    "exactly one"%name)

        return getattr(outputs[0], name)


    @property
    @typemap
    def arguments(self):
        '''
        List of all input variables of the Function that are not of type Parameter or Constant
        '''
        return super(Function, self).arguments()

    @property
    @typemap
    def attributes(self):
        '''
        List of the attributes of the function
        '''
        return super(Function, self).attributes()

    @typemap
    def clone(self, method, substitutions=None):
        '''
        Clones the function. The parameters of the Function are either cloned,
        shared or frozen as specified by the method argument and any variable
        substitutions requested are applied in the cloned Function instance.

        Args:
            method (:class:`CloneMethod`): one of

             * 'clone': the returned function gets its own copy of parameters (default)
             * 'share': the returned function shares its parameters with this function
             * 'freeze': parameters are cloned and made immutable (constant).

            substitutions (dict): a dictionary mapping variables in this
             function to variables in the cloned function

        Returns:
            :class:`~cntk.ops.functions.Function`: the cloned Function
        '''
        # C++ clone() can only clone composites. If we are not a composite, make it one using combine()
        if not self.is_composite:
            from cntk import combine
            #return combine([self]).clone(method, substitutions).root_function.arguments[0].owner
            # BUGBUG: This ^^ does not give me the correct .arguments, so we leave the extra combine() in for now.
            return combine([self]).clone(method, substitutions)

        method = getattr(cntk_py,
                'ParameterCloningMethod_' + CloneMethod(method).name.capitalize())
        substitutions = substitutions or {}
        if not isinstance(substitutions, dict):
            raise TypeError("Variable substitution map must be a dictionary")
        return super(Function, self).clone(method, substitutions)

    @property
    @typemap
    def constants(self):
        '''
        List of all `Constant` variables of this :class:`~cntk.ops.functions.Function`
        '''
        return super(Function, self).constants()

    def eval(self, arguments=None, device=None, as_numpy=True):
        '''
        Evaluate the node using the specified ``arguments`` as input.

        Args:
            arguments: maps variables to their input data. The interpretation depends on
             the input type:

               * dict: keys are input variable or names, and values are the input data.
                 See :meth:`~cntk.ops.functions.Function.forward` for details on passing
                 input data.
               * any other type: if node has an unique input, arguments is
                 mapped to this input.
             For nodes with more than one input, only dict is allowed.

             In both cases, every sample in the data will be interpreted
             as a new sequence.

             Sequences can be marked as continuations of the same sequence in
             the previous minibatch (that is the sequence in the same slot).
             There are two possibilities for this:

              * specifying arguments as a `tuple` where the first element is
                used as arguments and the second one will be used as a list
                of bools, denoting whether a sequence is a new one (`True`) or a
                continuation of the sequence in the same slot of the previous
                minibatch (`False`). This will be applied to all batches.
              * specifying arguments as a dictionary of variables to tuples
                where the first element is used as arguments and the second
                one will be used as a list of bools, denoting whether a sequence
                is a new one (`True`) or a continuation of the sequence in the
                same slot of the previous minibatch (`False`). This will be
                applied to all batches.

             Data should be either NumPy arrays or a
             :class:`~cntk.io.MinibatchData` instance.
            device (:class:`~cntk.device.DeviceDescriptor`): the device descriptor that
             contains the type and id of the device on which the computation is
             to be performed.
            as_numpy (bool): whether to return the result as a NumPy array. Default True.
             Specifying this as False returns a CNTK Value which avoids a
             costly conversion but returns a somewhat opaque object.

        Note:
             See :meth:`~cntk.ops.functions.Function.forward` for examples on
             passing input data.

        Returns:
           dict or NumPy Array: Dict with keys of ouput variable names and values of
           output variable. A single NumPy array if there is only one output value.
        '''

        _, output_map = self.forward(arguments, self.outputs, device=device, as_numpy=as_numpy)

        if len(output_map) > 1:
            return output_map
        else:
            return list(output_map.values())[0]


    @typemap
    def forward(self, arguments, outputs=None, keep_for_backward=None, device=None, as_numpy=True):
        '''
        Computes the values of speficied variables in ``outputs``, using values
        provided in ``arguments`` that correspond to each input `Variable` of
        the function (i.e. those that have ``is_input = True``).

        Example:
            >>> # Example of passing dense data
            >>> v = C.input_variable(shape=(3,))
            >>> f = C.reciprocal(v)
            >>> _, fv = f.forward({v:[[1, 2, 4]]})
            >>> list(fv.values())[0]
            array([[[ 1.  ,  0.5 ,  0.25]]], dtype=float32)

        Example:
            >>> # Passing sparse values as one-hot with a vocabulary size of 5
            >>> vocab_size = 5
            >>> v = C.input_variable(shape=(vocab_size,), is_sparse=True)
            >>> f = C.times(v, np.eye(vocab_size))
            >>> # Passing a batch of two sequences:
            >>> # 1st sequence: word 1
            >>> # 2nd sequence: words 2 and 4
            >>> batch = [[1],[2,4]]
            >>> sparse_batch = C.one_hot(batch, vocab_size)
            >>> _, fv = f.forward({v:sparse_batch})
            >>> list(fv.values())[0]
            [array([[ 0.,  1.,  0.,  0.,  0.]], dtype=float32),
             array([[ 0.,  0.,  1.,  0.,  0.], [ 0.,  0.,  0.,  0.,  1.]], dtype=float32)]

        Example:
            >>> # Doing the same, but with a CSR matrix from scipy.sparse
            >>> vocab_size = 5
            >>> from scipy.sparse import csr_matrix
            >>> v = C.input_variable(shape=(vocab_size,), is_sparse=True)
            >>> f = C.times(v, np.eye(vocab_size))
            >>> # Note that csr_matrix automatically uses a sparse representation underneath.
            >>> sparse_batch = [csr_matrix([[0,1,0,0,0]]), csr_matrix([[0,0,1,0,0], [0,0,0,0,1]])]
            >>> _, fv = f.forward({v:sparse_batch})
            >>> list(fv.values())[0]
            [array([[ 0.,  1.,  0.,  0.,  0.]], dtype=float32),
             array([[ 0.,  0.,  1.,  0.,  0.], [ 0.,  0.,  0.,  0.,  1.]], dtype=float32)]
            <BLANKLINE>
            >>> # Much more efficient, however, is to incrementally create CSR arrays.
            >>> # See https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html
            >>> # for more information.
            >>> def seq_to_csr_matrix(seq, vocab_size):
            ...     indptr = [0]
            ...     indices = []
            ...     data = []
            ...     for term_idx in seq:
            ...         indices.append(term_idx)
            ...         data.append(1)
            ...         indptr.append(len(indices))
            ...     return csr_matrix((data, indices, indptr), shape=(len(seq), vocab_size))
            >>> sparse_batch = [seq_to_csr_matrix(seq, vocab_size) for seq in batch]
            >>> _, fv = f.forward({v:sparse_batch})
            >>> list(fv.values())[0]
            [array([[ 0.,  1.,  0.,  0.,  0.]], dtype=float32),
             array([[ 0.,  0.,  1.,  0.,  0.], [ 0.,  0.,  0.,  0.,  1.]], dtype=float32)]


        Args:
            arguments: maps variables to their input data. The interpretation depends on
             the input type:

               * dict: keys are input variable or names, and values are the
                 input data. To specify a minibatch, provide a list of arrays.
                 The shape of each array must be compatible with the shape of
                 the dictionary key. If the array denotes a sequence then the
                 elements of the sequence are grouped along axis 0.
               * any other type: if node has an unique input, arguments is
                 mapped to this input.
             For nodes with more than one input, only dict is allowed.

             In both cases, every sample in the data will be interpreted
             as a new sequence.

             Sequences can be marked as continuations of the same sequence in
             the previous minibatch (that is the sequence in the same slot).
             There are two possibilities for this:

              * specifying arguments as a `tuple` where the first element is
                used as arguments and the second one will be used as a list
                of bools, denoting whether a sequence is a new one (`True`) or a
                continuation of the sequence in the same slot of the previous
                minibatch (`False`). This will be applied to all batches.
              * specifying arguments as a dictionary of variables to tuples
                where the first element is used as arguments and the second
                one will be used as a list of bools, denoting whether a sequence
                is a new one (`True`) or a continuation of the sequence in the
                same slot of the previous minibatch (`False`). This will be
                applied to all batches.

             Data should be either NumPy arrays or a
             :class:`~cntk.io.MinibatchData` instance.
            outputs (iterable, optional): outputs to fetch values for. If not
             set, all outputs of the function will be fetched.
            keep_for_backward (set, default `None`): the subset of the
             Function's output variables for which gradients shall be calculated
             in a subsequent backward call. If `None`, the returned state will
             be `None` and a subsequent call to :func:`backward` will not be
             possible.
            device (:class:`~cntk.device.DeviceDescriptor`, default `None`): the device
             descriptor that contains the type and id of the device on which the
             computation is. If `None`, the default device is used.
            as_numpy (bool): whether to return the result as a NumPy array. Default True.
             Specifying this as False returns a CNTK Value which avoids a
             costly conversion but returns a somewhat opaque object.

        Returns:
             A tuple (BackPropState, map of outputs to NumPy arrays). The
             BackPropState is a handle taken by :func:`backward`.
        '''
        if device is None:
            device = DeviceDescriptor.use_default_device()

        in_var_map = sanitize_var_map(self.arguments, arguments,
                                      None, device)
        if outputs is None:
            outputs = self.outputs

        output_map = {v: None for v in outputs}
        keep_for_backward = set(keep_for_backward or {})

        state = super(Function, self)._forward(in_var_map, output_map, device,
                                             keep_for_backward)
        if as_numpy:
            for k in output_map:
                output_map[k] = variable_value_to_seq(output_map[k], k)

        return state, output_map

    @typemap
    def backward(self, state, root_gradients, variables, as_numpy=True):
        '''
        Backpropagates supplied ``root_gradients`` for one or more of the output
        variables of the Function, to calculate gradients with respect to
        ``variables``. Formally, multiplies the values of ``root_gradients`` by
        the Jacobian of the Function and returns the subset of the output that
        corresponds to ``variables``.

        Example:
            >>> # compute the value and the derivative of the sigmoid at 0
            >>> v = C.input_variable(shape=(1,), needs_gradient=True)
            >>> f = C.sigmoid(v)
            >>> df, fv = f.forward({v:[[0]]}, [f.output], set([f.output]))
            >>> value = list(fv.values())[0]
            >>> grad = f.backward(df, {f.output: np.ones_like(value)}, set([v]))
            >>> value
            array([[[ 0.5]]], dtype=float32)
            >>> list(grad.values())[0]
            array([[[ 0.25]]], dtype=float32)

        Args:
            state (BackPropState): state obtained from a previous call to the
             func:`cntk.ops.Function.forward` method on this Function for the
             computation that this gradient backpropagation corresponds to.
            root_gradients (dict): the gradients that will be backpropagated
            variables (set): a list of input variables with respect to which
             the gradients have to be computed.
            as_numpy (bool): whether to return the gradients as a NumPy array. Default True.
             Specifying this as False returns a CNTK Value which avoids a
             costly conversion but returns a somewhat opaque object.

        Note:
             See :meth:`~cntk.ops.functions.Function.forward` for more examples
             on passing input data.

        Returns:
            dict: mapping of ``variables`` to NumPy arrays
        '''
        device = state.device()
        root_gradients = sanitize_var_map(self.outputs, root_gradients,
                                          None, device)

        var_gradients = dict((var, None) for var in variables)

        self._backward(state, root_gradients, var_gradients)

        if as_numpy:
            for var, value in var_gradients.items():
                var_gradients[var] = variable_value_to_seq(value, var)

        return var_gradients

    @typemap
    def grad(self, at, wrt=None, device=None, as_numpy=True):
        '''
        Computes the gradient of this Function at location ``at`` with respect to ``wrt``.
        The Function must have a single output.

        Example:
            >>> x = C.input_variable(shape=(1,), needs_gradient=True)
            >>> y = C.sqrt(x)
            >>> a = np.asarray([1,4,16],dtype=np.float32).reshape(3,1,1)
            >>> y.grad({x:a})
            array([[[ 0.5  ]],
            <BLANKLINE>
                   [[ 0.25 ]],
            <BLANKLINE>
                   [[ 0.125]]], dtype=float32)

        Args:
            at (dict) : mapping of the Function's arguments to values
            wrt (list optional): list of Variables with respect to which the
             gradient will be computed. If omitted, the gradients with
             respect to all arguments that need gradient will be computed. If a variable
             is repeated in this list, the gradient will be repeated
             in the output as a shallow copy.
            as_numpy (bool): whether to return the gradients as a NumPy array. Default True.
             Specifying this as False returns a CNTK Value which avoids a
             costly conversion but returns a somewhat opaque object.

        Returns:
            dict or NumPy Array: Dict with keys of ``wrt`` variables and gradient values of
            ``wrt`` variables. A single NumPy array if there is only one gradient value.
             Each element has the same shape as ``wrt`` including dynamic axes (such as the batch axis).
        '''

        if len(self.outputs) != 1 :
            raise InvalidArgumentException('function must return a single tensor')

        if wrt is None:
            wrt = [arg for arg in self.arguments if arg.needs_gradient]

        unique_wrt = set(wrt)
        output = [self.output]
        
        # Since we do not return the computed results and use them only to determine the shape
        # of the root gradients, we run the forward pass with as_numpy=False regardless of the
        # actual as_numpy setting passed to this function
        state, results = self.forward(at, output, set(output), device, as_numpy=False)
        ones = {self.output: np.ones(v.shape, self.output.dtype) for v in results.values()}
        grad_dict = self.backward(state, ones, unique_wrt, as_numpy)

        if len(grad_dict) > 1:
            return grad_dict
        else:
            return list(grad_dict.values())[0]

    @property
    @typemap
    def inputs(self):
        '''
        List of all input variables of this function.
        '''
        return super(Function, self).inputs(True)

    @property
    def name(self):
        '''
        Name of this function
        '''
        return super(Function, self).name()

    @name.setter
    def name(self, function_name):
        '''
        Sets the name of this Function.
        Setting the name of a Function is only allowed if the Function does not already have a name.
        Calling this method, when this Function already has a name, results in an exception.

        Args:
            function_name (`str`): name for this Function.
        '''
        super(Function, self).set_name(function_name)

    @property
    def op_name(self):
        '''
        Name of the operation that this Function performs
        '''
        return super(Function, self).op_name()

    @property
    @typemap
    def output(self):
        '''
        The single output variable if there is only one, or raises an exception.
        '''
        return super(Function, self).output()

    @property
    @typemap
    def outputs(self):
        '''
        List consisting of all output variables of this function.
        '''
        return super(Function, self).outputs()

    @property
    @typemap
    def parameters(self):
        '''
        List of all parameter variables of this function.
        '''
        return super(Function, self).parameters()

    @property
    @typemap
    def placeholders(self):
        '''
        List of all placeholders variables of this function.
        '''
        return super(Function, self).placeholders()

    @property
    @typemap
    def root_function(self):
        '''
        The primitive function at the root of the graph of functions underlying this function.
        '''
        return super(Function, self).root_function()

    @property
    def is_primitive(self):
        '''
        Returns a boolean indicating if this Function is a primitive Function.
        A primitive Function is the lowest level building block for composite Function
        graphs and is either a CNTK built-in operator, a composite Function encapsulated
        as a Block or a user-defined Function
        '''
        return super(Function, self).is_primitive()

    @property
    def is_composite(self):
        '''
        Returns a boolean indicating if this Function is a composite Function.
        A composite Function is a Function that is composed of primitive Functions.
        '''
        return super(Function, self).is_composite()

    @property
    def is_block(self):
        '''
        Returns a boolean indicating if this Function is a block function which is basically
        a composite encapsulated as an opaque block which appears as a primitive during
        traversing the graph of Functions that this block is part of.
        '''
        return super(Function, self).is_block()

    @property
    @typemap
    def block_root(self):
        '''
        Returns the root of the Function graph underlying this block Function.
        Throws an exception if this is not a block Function.
        '''
        return super(Function, self).block_root()

    @property
    @typemap
    def block_arguments_mapping(self):
        '''
        Returns the mapping from the arguments of the composite underlying this block function
        to the Variables that they are bound to in the outer graph of Functions that this
        block Function is part of.
        '''
        return super(Function, self).block_arguments_mapping()

    @property
    @typemap
    def uid(self):
        '''
        The internally generated unique name of the function.
        '''
        return super(Function, self).uid()

    @typemap
    def replace_placeholders(self, substitutions):
        '''
        In-place replace specified placeholders in the Function graph with the
        specified replacements in the map.

        Args:
            substitutions (dict): map from placeholder to variables

        Returns:
            :class:`Function`: itself
        '''
        substitutions = substitutions or {}
        if not isinstance(substitutions, dict):
            raise TypeError("Variable substitution map must be a dictionary")
        return super(Function, self).replace_placeholders(substitutions)

    @typemap
    def replace_placeholder(self, substitution):
        '''
        In-place replace the only placeholder in the function graph with the
        specified substitution.

        Args:
            substitution (:class:`~cntk.ops.variables.Variable`): the variable
             that will replace the placeholder

        Returns:
            :class:`Function`: itself

        :raises ExceptionType: when the function has multiple placeholders.
        '''
        return super(Function, self).replace_placeholder(substitution)

    @typemap
    def find_all_with_name(self, name):
        '''
        Returns a list of primitive function with ``name`` in the graph
        starting from this node. Throws an exception if ``name`` occurs
        multiple times. If you expect only one function to be returned, use
        :func:`find_by_name`.

        Example:
            >>> a = C.input_variable(shape=1, name='i')
            >>> b = C.input_variable(shape=1, name='i')
            >>> c = C.plus(a, b, name='c')
            >>> len(c.find_all_with_name('i'))
            2
            >>> c.find_all_with_name('z')
            []

        Args:
            name (str): names to look for

        Returns:
            list of :class:`Function` objects matching ``name``

        See also:
            :func:`find_by_name`
        '''
        from .. import graph
        return graph.find_all_with_name(self, name)

    # TODO have a better name for combine() in this case
    @typemap
    def find_by_name(self, name):
        '''
        Returns a primitive function with ``name`` in the graph starting from
        this node. Throws an exception if ``name`` occurs multiple times. If
        you expect multiple functions to be returned, use
        :func:`find_all_with_name`.

        Example:
            >>> a = C.input_variable(shape=1, name='a')
            >>> b = C.input_variable(shape=1, name='b')
            >>> c = C.plus(a, b, name='c')
            >>> print(c.find_by_name('b').name)
            b
            >>> c.find_by_name('z') is None
            True

            If you need a full function out of it that can be evaluated, you
            need to upcast it (currently done via combine):

            >>> d = c * 5
            >>> C.combine([d.find_by_name('c')]).eval({a:[[1]], b:[[2]]})
            array([[[ 3.]]], dtype=float32)

        Args:
            name (str): names to look for

        Returns:
            :class:`Function` object matching ``name``

        See also:
            :func:`find_all_with_name`
        '''
        from .. import graph
        return graph.find_by_name(self, name)

    @typemap
    def save(self, filename):
        '''
        Save this function graph into a model file using protobuf-based
        serialization.

        Use distributed.Communicator.is_main() to gate your call to save()
        in distributed environment.

        Args:
            filename (str): model path
        '''
        return super(Function, self).save_model(filename)

    def save_model(self, filename): # legacy name
        import warnings
        warnings.warn('This will be removed in future versions. Please use '
                'save(...) instead', DeprecationWarning)
        return self.save(filename)

    @typemap
    def restore(self, filename):
        '''
        Restore the models parameters (in-place) from a saved model file

        Args:
            filename (str): saved model path

        Returns:
            `None`: this method only has the side-effect of loading the model parameters from the file
        '''
        return super(Function, self).restore_model(filename)

    def restore_model(self, filename): # legacy name
        import warnings
        warnings.warn('This will be removed in future versions. Please use '
                'restore(...) instead', DeprecationWarning)
        return self.restore(filename)

    @staticmethod
    @typemap
    def load(filename, device=None):
        '''
        Load the model in ``filename``, that has been saved using
        :func:`~cntk.ops.functions.Function.save`.

        Args:
            filename (str): filename to load the model from
            device (:class:`~cntk.device.DeviceDescriptor`, default is the default device):
             instance of DeviceDescriptor

        Returns:
            root node
        '''
        if not device:
            device = DeviceDescriptor.use_default_device()
        return cntk_py.Function.load_model(filename, device)

@typemap
def load_model(filename, device=None):
    '''
    Alias for :func:`~cntk.ops.functions.Function.load`.
    '''
    return Function.load(filename, device)

@typemap
def save_model(model, filename): # legacy name
    import warnings
    warnings.warn('This will be removed in future versions. Please use '
            'model.save(...) instead', DeprecationWarning)
    return model.save(filename)


class UserFunction(Function):
    '''
    Base class of all user extension functions.

    If it has only one output, one can invoke Variable methods on it, which it
    will relay to its only output.

    Args:
        inputs (list): inputs to this function
        as_numpy (bool, optional): whether the data should be automatically
         converted from and to NumPy. Defaults to True. Specifying this as
         `False` passes the data as CNTK Value objects.
        name (str): name of this function
    '''
    def __init__(self, inputs, as_numpy=True, name=''):
        super(UserFunction, self).__init__(inputs, name)
        self.as_numpy = as_numpy

        # Since the state will frequently not be used, we cache the None-state
        # to speed up.
        self._none_state =  cntk_py.UserBackPropState(self,
                DeviceDescriptor.cpu_device(), None)

        # Memory management for user defined functions has to be controlled by
        # the C++ side. For more information:
        # http://www.swig.org/Doc3.0/Python.html#Python_nn35
        self.__disown__()


    def _forward(self, arguments, outputs, device=None, outputs_to_retain=None):
        '''
        Computes the values of speficied variables in ``outputs``, using values
        provided in ``arguments`` that correspond to each input `Variable` of
        the function whose ``is_input`` is `True`.

        This function calls :func:`forward`, which is to be implemented by the
        user.

        Args:
            arguments (tuple): Value objects of the Function's input
            outputs (iterable): outputs to fetch values for.
            device (:class:`~cntk.device.DeviceDescriptor`, default `None`): the device
             descriptor that contains the type and id of the device on which the
             computation is. If `None`, the default device is used.

        Returns:
             A BackPropState instance, which is used by :func:`backward`.
        '''
        if self.as_numpy:
            arguments = tuple(variable_value_to_seq(v, self.inputs[i]) for i, v in enumerate(arguments))

        map_if_possible(outputs)
        map_if_possible(outputs_to_retain)

        args = arguments if len(arguments)>1 else arguments[0]

        if len(outputs) <= 1:
            state, result = self.forward(args, device, outputs_to_retain)
            for k in outputs:
                outputs[k] = result
        else:
            state = self.forward(args, outputs, device, outputs_to_retain)

        if state is None:
            state = self._none_state
        elif not isinstance(state, cntk_py.BackPropState):
            state = cntk_py.UserBackPropState(self, device, state)

        if self.as_numpy:
            for k,v in outputs.items():
                if v is None:
                    raise ValueError('not all outputs have been provided')

                # FIXME: seq_starts
                outputs[k] = sanitize_batch(k, v, None, device)

        return state, outputs

    def _backward(self, state, root_gradients, variables):
        '''
        Backpropagates supplied ``root_gradients`` for one or more of the output
        variables of the Function, to calculate gradients with respect to
        ``variables``. Formally, multiplies the values of ``root_gradients`` by
        the Jacobian of the Function and returns the subset of the output that
        corresponds to ``variables``.

        This function calls :func:`backward`, which is to be implemented by the
        user.

        Args:
            state (BackPropState): state obtained from a previous call to the
             func:`cntk.ops.Function.forward` method on this Function for the
             computation that this gradient backpropagation corresponds to.
            root_gradients (dict): the gradients that will be backpropagated
            variables (set): a list of input variables with respect to which
             the gradients have to be computed.

        Returns:
            dict: mapping of ``variables`` to NumPy arrays
        '''
        device = state.device()

        if self.as_numpy:
            for v in root_gradients:
                root_gradients[v] = variable_value_to_seq(root_gradients[v], v)

            state = cntk_py.UserBackPropState.data(state)

        else:
            if not isinstance(state, cntk_py.BackPropState):
                if state is None:
                    state = self._none_state
                else:
                    raise ValueError('if as_numpy=False, state must be of '
                            'type BackPropState')

        map_if_possible(variables)

        if len(variables)>1:
            self.backward(state, root_gradients, variables)
        else:
            for rg in root_gradients.values():
                break
            result = self.backward(state, rg)
            for k in variables:
                variables[k] = result

        if self.as_numpy:
            for k,v in variables.items():
                if v is None:
                    raise ValueError('gradients were not provided for all variables')

                variables[k] = sanitize_batch(k, v, None, device)

    def _infer_outputs(self, outputs):
        outputs.extend(self.infer_outputs())

    def infer_outputs(self):
        '''
        Returns a list of all output variables this user-defined function
        outputs.

        Output variables are created by
        :meth:`~cntk.ops.functions.output_variable`.
        '''
        raise NotImplementedError('infer_outputs has to be overwritten')

    def clone(self, cloned_inputs):
        '''
        Creates a clone of this user-defined function.

        Args:
            cloned_inputs: list of cloned inputs to the new user-defined
             Function clone to be created.

        Returns:
            A cloned instance of this user-defined function.
        '''
        raise NotImplementedError('clone has to be overwritten')

    def op_name(self):
        '''
        Returns the operator name.
        '''
        return 'UserFunction'
