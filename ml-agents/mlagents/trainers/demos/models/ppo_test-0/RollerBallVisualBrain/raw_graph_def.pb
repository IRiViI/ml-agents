
C
global_step/initial_valueConst*
value	B : *
dtype0
W
global_step
VariableV2*
shape: *
shared_name *
dtype0*
	container 
�
global_step/AssignAssignglobal_stepglobal_step/initial_value*
use_locking(*
T0*
_class
loc:@global_step*
validate_shape(
R
global_step/readIdentityglobal_step*
T0*
_class
loc:@global_step
;
steps_to_incrementPlaceholder*
shape: *
dtype0
9
AddAddglobal_step/readsteps_to_increment*
T0
t
AssignAssignglobal_stepAdd*
use_locking(*
T0*
_class
loc:@global_step*
validate_shape(
5

batch_sizePlaceholder*
shape:*
dtype0
:
sequence_lengthPlaceholder*
shape:*
dtype0
;
masksPlaceholder*
shape:���������*
dtype0
;
CastCastmasks*

SrcT0*
Truncate( *

DstT0
M
#is_continuous_control/initial_valueConst*
value	B : *
dtype0
a
is_continuous_control
VariableV2*
shape: *
shared_name *
dtype0*
	container 
�
is_continuous_control/AssignAssignis_continuous_control#is_continuous_control/initial_value*
use_locking(*
T0*(
_class
loc:@is_continuous_control*
validate_shape(
p
is_continuous_control/readIdentityis_continuous_control*
T0*(
_class
loc:@is_continuous_control
F
version_number/initial_valueConst*
value	B :*
dtype0
Z
version_number
VariableV2*
shape: *
shared_name *
dtype0*
	container 
�
version_number/AssignAssignversion_numberversion_number/initial_value*
use_locking(*
T0*!
_class
loc:@version_number*
validate_shape(
[
version_number/readIdentityversion_number*
T0*!
_class
loc:@version_number
D
memory_size/initial_valueConst*
value
B :�*
dtype0
W
memory_size
VariableV2*
shape: *
shared_name *
dtype0*
	container 
�
memory_size/AssignAssignmemory_sizememory_size/initial_value*
use_locking(*
T0*
_class
loc:@memory_size*
validate_shape(
R
memory_size/readIdentitymemory_size*
T0*
_class
loc:@memory_size
K
!action_output_shape/initial_valueConst*
value	B :*
dtype0
_
action_output_shape
VariableV2*
shape: *
shared_name *
dtype0*
	container 
�
action_output_shape/AssignAssignaction_output_shape!action_output_shape/initial_value*
use_locking(*
T0*&
_class
loc:@action_output_shape*
validate_shape(
j
action_output_shape/readIdentityaction_output_shape*
T0*&
_class
loc:@action_output_shape
V
visual_observation_0Placeholder*$
shape:���������TT*
dtype0
J
vector_observationPlaceholder*
shape:��������� *
dtype0
�
Dmain_graph_0_encoder0/conv_0/kernel/Initializer/random_uniform/shapeConst*6
_class,
*(loc:@main_graph_0_encoder0/conv_0/kernel*%
valueB"             *
dtype0
�
Bmain_graph_0_encoder0/conv_0/kernel/Initializer/random_uniform/minConst*6
_class,
*(loc:@main_graph_0_encoder0/conv_0/kernel*
valueB
 *��S�*
dtype0
�
Bmain_graph_0_encoder0/conv_0/kernel/Initializer/random_uniform/maxConst*6
_class,
*(loc:@main_graph_0_encoder0/conv_0/kernel*
valueB
 *��S=*
dtype0
�
Lmain_graph_0_encoder0/conv_0/kernel/Initializer/random_uniform/RandomUniformRandomUniformDmain_graph_0_encoder0/conv_0/kernel/Initializer/random_uniform/shape*
seed�8*
T0*6
_class,
*(loc:@main_graph_0_encoder0/conv_0/kernel*
dtype0*
seed2 
�
Bmain_graph_0_encoder0/conv_0/kernel/Initializer/random_uniform/subSubBmain_graph_0_encoder0/conv_0/kernel/Initializer/random_uniform/maxBmain_graph_0_encoder0/conv_0/kernel/Initializer/random_uniform/min*
T0*6
_class,
*(loc:@main_graph_0_encoder0/conv_0/kernel
�
Bmain_graph_0_encoder0/conv_0/kernel/Initializer/random_uniform/mulMulLmain_graph_0_encoder0/conv_0/kernel/Initializer/random_uniform/RandomUniformBmain_graph_0_encoder0/conv_0/kernel/Initializer/random_uniform/sub*
T0*6
_class,
*(loc:@main_graph_0_encoder0/conv_0/kernel
�
>main_graph_0_encoder0/conv_0/kernel/Initializer/random_uniformAddBmain_graph_0_encoder0/conv_0/kernel/Initializer/random_uniform/mulBmain_graph_0_encoder0/conv_0/kernel/Initializer/random_uniform/min*
T0*6
_class,
*(loc:@main_graph_0_encoder0/conv_0/kernel
�
#main_graph_0_encoder0/conv_0/kernel
VariableV2*
shape: *
shared_name *6
_class,
*(loc:@main_graph_0_encoder0/conv_0/kernel*
dtype0*
	container 
�
*main_graph_0_encoder0/conv_0/kernel/AssignAssign#main_graph_0_encoder0/conv_0/kernel>main_graph_0_encoder0/conv_0/kernel/Initializer/random_uniform*
use_locking(*
T0*6
_class,
*(loc:@main_graph_0_encoder0/conv_0/kernel*
validate_shape(
�
(main_graph_0_encoder0/conv_0/kernel/readIdentity#main_graph_0_encoder0/conv_0/kernel*
T0*6
_class,
*(loc:@main_graph_0_encoder0/conv_0/kernel
�
3main_graph_0_encoder0/conv_0/bias/Initializer/zerosConst*4
_class*
(&loc:@main_graph_0_encoder0/conv_0/bias*
valueB *    *
dtype0
�
!main_graph_0_encoder0/conv_0/bias
VariableV2*
shape: *
shared_name *4
_class*
(&loc:@main_graph_0_encoder0/conv_0/bias*
dtype0*
	container 
�
(main_graph_0_encoder0/conv_0/bias/AssignAssign!main_graph_0_encoder0/conv_0/bias3main_graph_0_encoder0/conv_0/bias/Initializer/zeros*
use_locking(*
T0*4
_class*
(&loc:@main_graph_0_encoder0/conv_0/bias*
validate_shape(
�
&main_graph_0_encoder0/conv_0/bias/readIdentity!main_graph_0_encoder0/conv_0/bias*
T0*4
_class*
(&loc:@main_graph_0_encoder0/conv_0/bias
_
*main_graph_0_encoder0/conv_0/dilation_rateConst*
valueB"      *
dtype0
�
#main_graph_0_encoder0/conv_0/Conv2DConv2Dvisual_observation_0(main_graph_0_encoder0/conv_0/kernel/read*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingVALID
�
$main_graph_0_encoder0/conv_0/BiasAddBiasAdd#main_graph_0_encoder0/conv_0/Conv2D&main_graph_0_encoder0/conv_0/bias/read*
T0*
data_formatNHWC
V
 main_graph_0_encoder0/conv_0/EluElu$main_graph_0_encoder0/conv_0/BiasAdd*
T0
�
Dmain_graph_0_encoder0/conv_1/kernel/Initializer/random_uniform/shapeConst*6
_class,
*(loc:@main_graph_0_encoder0/conv_1/kernel*%
valueB"              *
dtype0
�
Bmain_graph_0_encoder0/conv_1/kernel/Initializer/random_uniform/minConst*6
_class,
*(loc:@main_graph_0_encoder0/conv_1/kernel*
valueB
 *qĜ�*
dtype0
�
Bmain_graph_0_encoder0/conv_1/kernel/Initializer/random_uniform/maxConst*6
_class,
*(loc:@main_graph_0_encoder0/conv_1/kernel*
valueB
 *qĜ=*
dtype0
�
Lmain_graph_0_encoder0/conv_1/kernel/Initializer/random_uniform/RandomUniformRandomUniformDmain_graph_0_encoder0/conv_1/kernel/Initializer/random_uniform/shape*
seed�8*
T0*6
_class,
*(loc:@main_graph_0_encoder0/conv_1/kernel*
dtype0*
seed22
�
Bmain_graph_0_encoder0/conv_1/kernel/Initializer/random_uniform/subSubBmain_graph_0_encoder0/conv_1/kernel/Initializer/random_uniform/maxBmain_graph_0_encoder0/conv_1/kernel/Initializer/random_uniform/min*
T0*6
_class,
*(loc:@main_graph_0_encoder0/conv_1/kernel
�
Bmain_graph_0_encoder0/conv_1/kernel/Initializer/random_uniform/mulMulLmain_graph_0_encoder0/conv_1/kernel/Initializer/random_uniform/RandomUniformBmain_graph_0_encoder0/conv_1/kernel/Initializer/random_uniform/sub*
T0*6
_class,
*(loc:@main_graph_0_encoder0/conv_1/kernel
�
>main_graph_0_encoder0/conv_1/kernel/Initializer/random_uniformAddBmain_graph_0_encoder0/conv_1/kernel/Initializer/random_uniform/mulBmain_graph_0_encoder0/conv_1/kernel/Initializer/random_uniform/min*
T0*6
_class,
*(loc:@main_graph_0_encoder0/conv_1/kernel
�
#main_graph_0_encoder0/conv_1/kernel
VariableV2*
shape:  *
shared_name *6
_class,
*(loc:@main_graph_0_encoder0/conv_1/kernel*
dtype0*
	container 
�
*main_graph_0_encoder0/conv_1/kernel/AssignAssign#main_graph_0_encoder0/conv_1/kernel>main_graph_0_encoder0/conv_1/kernel/Initializer/random_uniform*
use_locking(*
T0*6
_class,
*(loc:@main_graph_0_encoder0/conv_1/kernel*
validate_shape(
�
(main_graph_0_encoder0/conv_1/kernel/readIdentity#main_graph_0_encoder0/conv_1/kernel*
T0*6
_class,
*(loc:@main_graph_0_encoder0/conv_1/kernel
�
3main_graph_0_encoder0/conv_1/bias/Initializer/zerosConst*4
_class*
(&loc:@main_graph_0_encoder0/conv_1/bias*
valueB *    *
dtype0
�
!main_graph_0_encoder0/conv_1/bias
VariableV2*
shape: *
shared_name *4
_class*
(&loc:@main_graph_0_encoder0/conv_1/bias*
dtype0*
	container 
�
(main_graph_0_encoder0/conv_1/bias/AssignAssign!main_graph_0_encoder0/conv_1/bias3main_graph_0_encoder0/conv_1/bias/Initializer/zeros*
use_locking(*
T0*4
_class*
(&loc:@main_graph_0_encoder0/conv_1/bias*
validate_shape(
�
&main_graph_0_encoder0/conv_1/bias/readIdentity!main_graph_0_encoder0/conv_1/bias*
T0*4
_class*
(&loc:@main_graph_0_encoder0/conv_1/bias
_
*main_graph_0_encoder0/conv_1/dilation_rateConst*
valueB"      *
dtype0
�
#main_graph_0_encoder0/conv_1/Conv2DConv2D main_graph_0_encoder0/conv_0/Elu(main_graph_0_encoder0/conv_1/kernel/read*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingVALID
�
$main_graph_0_encoder0/conv_1/BiasAddBiasAdd#main_graph_0_encoder0/conv_1/Conv2D&main_graph_0_encoder0/conv_1/bias/read*
T0*
data_formatNHWC
V
 main_graph_0_encoder0/conv_1/EluElu$main_graph_0_encoder0/conv_1/BiasAdd*
T0
o
+main_graph_0_encoder0/Flatten/flatten/ShapeShape main_graph_0_encoder0/conv_1/Elu*
T0*
out_type0
g
9main_graph_0_encoder0/Flatten/flatten/strided_slice/stackConst*
valueB: *
dtype0
i
;main_graph_0_encoder0/Flatten/flatten/strided_slice/stack_1Const*
valueB:*
dtype0
i
;main_graph_0_encoder0/Flatten/flatten/strided_slice/stack_2Const*
valueB:*
dtype0
�
3main_graph_0_encoder0/Flatten/flatten/strided_sliceStridedSlice+main_graph_0_encoder0/Flatten/flatten/Shape9main_graph_0_encoder0/Flatten/flatten/strided_slice/stack;main_graph_0_encoder0/Flatten/flatten/strided_slice/stack_1;main_graph_0_encoder0/Flatten/flatten/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask 
h
5main_graph_0_encoder0/Flatten/flatten/Reshape/shape/1Const*
valueB :
���������*
dtype0
�
3main_graph_0_encoder0/Flatten/flatten/Reshape/shapePack3main_graph_0_encoder0/Flatten/flatten/strided_slice5main_graph_0_encoder0/Flatten/flatten/Reshape/shape/1*
T0*

axis *
N
�
-main_graph_0_encoder0/Flatten/flatten/ReshapeReshape main_graph_0_encoder0/conv_1/Elu3main_graph_0_encoder0/Flatten/flatten/Reshape/shape*
T0*
Tshape0
�
lmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/kernel/Initializer/truncated_normal/shapeConst*\
_classR
PNloc:@main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/kernel*
valueB" 
      *
dtype0
�
kmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/kernel/Initializer/truncated_normal/meanConst*\
_classR
PNloc:@main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/kernel*
valueB
 *    *
dtype0
�
mmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/kernel/Initializer/truncated_normal/stddevConst*\
_classR
PNloc:@main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/kernel*
valueB
 *v�<*
dtype0
�
vmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormallmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/kernel/Initializer/truncated_normal/shape*
seed�8*
T0*\
_classR
PNloc:@main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/kernel*
dtype0*
seed2L
�
jmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/kernel/Initializer/truncated_normal/mulMulvmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/kernel/Initializer/truncated_normal/TruncatedNormalmmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/kernel/Initializer/truncated_normal/stddev*
T0*\
_classR
PNloc:@main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/kernel
�
fmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/kernel/Initializer/truncated_normalAddjmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/kernel/Initializer/truncated_normal/mulkmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/kernel/Initializer/truncated_normal/mean*
T0*\
_classR
PNloc:@main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/kernel
�
Imain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/kernel
VariableV2*
shape:	� *
shared_name *\
_classR
PNloc:@main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/kernel*
dtype0*
	container 
�
Pmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/kernel/AssignAssignImain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/kernelfmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/kernel/Initializer/truncated_normal*
use_locking(*
T0*\
_classR
PNloc:@main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/kernel*
validate_shape(
�
Nmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/kernel/readIdentityImain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/kernel*
T0*\
_classR
PNloc:@main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/kernel
�
Ymain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/bias/Initializer/zerosConst*Z
_classP
NLloc:@main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/bias*
valueB *    *
dtype0
�
Gmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/bias
VariableV2*
shape: *
shared_name *Z
_classP
NLloc:@main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/bias*
dtype0*
	container 
�
Nmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/bias/AssignAssignGmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/biasYmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/bias/Initializer/zeros*
use_locking(*
T0*Z
_classP
NLloc:@main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/bias*
validate_shape(
�
Lmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/bias/readIdentityGmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/bias*
T0*Z
_classP
NLloc:@main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/bias
�
Imain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/MatMulMatMul-main_graph_0_encoder0/Flatten/flatten/ReshapeNmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/kernel/read*
transpose_b( *
T0*
transpose_a( 
�
Jmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/BiasAddBiasAddImain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/MatMulLmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/bias/read*
T0*
data_formatNHWC
�
Jmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/SigmoidSigmoidJmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/BiasAdd*
T0
�
Fmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/MulMulJmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/BiasAddJmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/Sigmoid*
T0
�
lmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/kernel/Initializer/truncated_normal/shapeConst*\
_classR
PNloc:@main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/kernel*
valueB"        *
dtype0
�
kmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/kernel/Initializer/truncated_normal/meanConst*\
_classR
PNloc:@main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/kernel*
valueB
 *    *
dtype0
�
mmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/kernel/Initializer/truncated_normal/stddevConst*\
_classR
PNloc:@main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/kernel*
valueB
 *�dN>*
dtype0
�
vmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormallmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/kernel/Initializer/truncated_normal/shape*
seed�8*
T0*\
_classR
PNloc:@main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/kernel*
dtype0*
seed2]
�
jmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/kernel/Initializer/truncated_normal/mulMulvmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/kernel/Initializer/truncated_normal/TruncatedNormalmmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/kernel/Initializer/truncated_normal/stddev*
T0*\
_classR
PNloc:@main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/kernel
�
fmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/kernel/Initializer/truncated_normalAddjmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/kernel/Initializer/truncated_normal/mulkmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/kernel/Initializer/truncated_normal/mean*
T0*\
_classR
PNloc:@main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/kernel
�
Imain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/kernel
VariableV2*
shape
:  *
shared_name *\
_classR
PNloc:@main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/kernel*
dtype0*
	container 
�
Pmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/kernel/AssignAssignImain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/kernelfmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/kernel/Initializer/truncated_normal*
use_locking(*
T0*\
_classR
PNloc:@main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/kernel*
validate_shape(
�
Nmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/kernel/readIdentityImain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/kernel*
T0*\
_classR
PNloc:@main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/kernel
�
Ymain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/bias/Initializer/zerosConst*Z
_classP
NLloc:@main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/bias*
valueB *    *
dtype0
�
Gmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/bias
VariableV2*
shape: *
shared_name *Z
_classP
NLloc:@main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/bias*
dtype0*
	container 
�
Nmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/bias/AssignAssignGmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/biasYmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/bias/Initializer/zeros*
use_locking(*
T0*Z
_classP
NLloc:@main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/bias*
validate_shape(
�
Lmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/bias/readIdentityGmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/bias*
T0*Z
_classP
NLloc:@main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/bias
�
Imain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/MatMulMatMulFmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/MulNmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/kernel/read*
transpose_b( *
T0*
transpose_a( 
�
Jmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/BiasAddBiasAddImain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/MatMulLmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/bias/read*
T0*
data_formatNHWC
�
Jmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/SigmoidSigmoidJmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/BiasAdd*
T0
�
Fmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/MulMulJmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/BiasAddJmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/Sigmoid*
T0
;
concat/concat_dimConst*
value	B :*
dtype0
j
concat/concatIdentityFmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/Mul*
T0
E
prev_actionPlaceholder*
shape:���������*
dtype0
H
strided_slice/stackConst*
valueB"        *
dtype0
J
strided_slice/stack_1Const*
valueB"       *
dtype0
J
strided_slice/stack_2Const*
valueB"      *
dtype0
�
strided_sliceStridedSliceprev_actionstrided_slice/stackstrided_slice/stack_1strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask
=
one_hot/on_valueConst*
valueB
 *  �?*
dtype0
>
one_hot/off_valueConst*
valueB
 *    *
dtype0
7
one_hot/depthConst*
value	B :*
dtype0
|
one_hotOneHotstrided_sliceone_hot/depthone_hot/on_valueone_hot/off_value*
T0*
TI0*
axis���������
=
concat_1/concat_dimConst*
value	B :*
dtype0
-
concat_1/concatIdentityone_hot*
T0
7
concat_2/axisConst*
value	B :*
dtype0
a
concat_2ConcatV2concat/concatconcat_1/concatconcat_2/axis*

Tidx0*
T0*
N
G
recurrent_inPlaceholder*
shape:����������*
dtype0
B
Reshape/shape/0Const*
valueB :
���������*
dtype0
9
Reshape/shape/2Const*
value	B :$*
dtype0
f
Reshape/shapePackReshape/shape/0sequence_lengthReshape/shape/2*
T0*

axis *
N
B
ReshapeReshapeconcat_2Reshape/shape*
T0*
Tshape0
J
strided_slice_1/stackConst*
valueB"        *
dtype0
L
strided_slice_1/stack_1Const*
valueB"        *
dtype0
L
strided_slice_1/stack_2Const*
valueB"      *
dtype0
�
strided_slice_1StridedSlicerecurrent_instrided_slice_1/stackstrided_slice_1/stack_1strided_slice_1/stack_2*
T0*
Index0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask
D
Reshape_1/shapeConst*
valueB"����   *
dtype0
M
	Reshape_1Reshapestrided_slice_1Reshape_1/shape*
T0*
Tshape0
M
lstm/strided_slice/stackConst*
valueB"        *
dtype0
O
lstm/strided_slice/stack_1Const*
valueB"    �   *
dtype0
O
lstm/strided_slice/stack_2Const*
valueB"      *
dtype0
�
lstm/strided_sliceStridedSlice	Reshape_1lstm/strided_slice/stacklstm/strided_slice/stack_1lstm/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask
O
lstm/strided_slice_1/stackConst*
valueB"    �   *
dtype0
Q
lstm/strided_slice_1/stack_1Const*
valueB"        *
dtype0
Q
lstm/strided_slice_1/stack_2Const*
valueB"      *
dtype0
�
lstm/strided_slice_1StridedSlice	Reshape_1lstm/strided_slice_1/stacklstm/strided_slice_1/stack_1lstm/strided_slice_1/stack_2*
T0*
Index0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask
7
lstm/rnn/RankConst*
value	B :*
dtype0
>
lstm/rnn/range/startConst*
value	B :*
dtype0
>
lstm/rnn/range/deltaConst*
value	B :*
dtype0
^
lstm/rnn/rangeRangelstm/rnn/range/startlstm/rnn/Ranklstm/rnn/range/delta*

Tidx0
M
lstm/rnn/concat/values_0Const*
valueB"       *
dtype0
>
lstm/rnn/concat/axisConst*
value	B : *
dtype0
y
lstm/rnn/concatConcatV2lstm/rnn/concat/values_0lstm/rnn/rangelstm/rnn/concat/axis*

Tidx0*
T0*
N
O
lstm/rnn/transpose	TransposeReshapelstm/rnn/concat*
Tperm0*
T0
D
lstm/rnn/ShapeShapelstm/rnn/transpose*
T0*
out_type0
J
lstm/rnn/strided_slice/stackConst*
valueB:*
dtype0
L
lstm/rnn/strided_slice/stack_1Const*
valueB:*
dtype0
L
lstm/rnn/strided_slice/stack_2Const*
valueB:*
dtype0
�
lstm/rnn/strided_sliceStridedSlicelstm/rnn/Shapelstm/rnn/strided_slice/stacklstm/rnn/strided_slice/stack_1lstm/rnn/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask 
F
lstm/rnn/Shape_1Shapelstm/rnn/transpose*
T0*
out_type0
L
lstm/rnn/strided_slice_1/stackConst*
valueB: *
dtype0
N
 lstm/rnn/strided_slice_1/stack_1Const*
valueB:*
dtype0
N
 lstm/rnn/strided_slice_1/stack_2Const*
valueB:*
dtype0
�
lstm/rnn/strided_slice_1StridedSlicelstm/rnn/Shape_1lstm/rnn/strided_slice_1/stack lstm/rnn/strided_slice_1/stack_1 lstm/rnn/strided_slice_1/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask 
F
lstm/rnn/Shape_2Shapelstm/rnn/transpose*
T0*
out_type0
L
lstm/rnn/strided_slice_2/stackConst*
valueB:*
dtype0
N
 lstm/rnn/strided_slice_2/stack_1Const*
valueB:*
dtype0
N
 lstm/rnn/strided_slice_2/stack_2Const*
valueB:*
dtype0
�
lstm/rnn/strided_slice_2StridedSlicelstm/rnn/Shape_2lstm/rnn/strided_slice_2/stack lstm/rnn/strided_slice_2/stack_1 lstm/rnn/strided_slice_2/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask 
A
lstm/rnn/ExpandDims/dimConst*
value	B : *
dtype0
i
lstm/rnn/ExpandDims
ExpandDimslstm/rnn/strided_slice_2lstm/rnn/ExpandDims/dim*

Tdim0*
T0
=
lstm/rnn/ConstConst*
valueB:�*
dtype0
@
lstm/rnn/concat_1/axisConst*
value	B : *
dtype0
x
lstm/rnn/concat_1ConcatV2lstm/rnn/ExpandDimslstm/rnn/Constlstm/rnn/concat_1/axis*

Tidx0*
T0*
N
A
lstm/rnn/zeros/ConstConst*
valueB
 *    *
dtype0
Z
lstm/rnn/zerosFilllstm/rnn/concat_1lstm/rnn/zeros/Const*
T0*

index_type0
7
lstm/rnn/timeConst*
value	B : *
dtype0
�
lstm/rnn/TensorArrayTensorArrayV3lstm/rnn/strided_slice_1*%
element_shape:����������*
dynamic_size( *
clear_after_read(*
identical_element_shapes(*4
tensor_array_namelstm/rnn/dynamic_rnn/output_0*
dtype0
�
lstm/rnn/TensorArray_1TensorArrayV3lstm/rnn/strided_slice_1*$
element_shape:���������$*
dynamic_size( *
clear_after_read(*
identical_element_shapes(*3
tensor_array_namelstm/rnn/dynamic_rnn/input_0*
dtype0
W
!lstm/rnn/TensorArrayUnstack/ShapeShapelstm/rnn/transpose*
T0*
out_type0
]
/lstm/rnn/TensorArrayUnstack/strided_slice/stackConst*
valueB: *
dtype0
_
1lstm/rnn/TensorArrayUnstack/strided_slice/stack_1Const*
valueB:*
dtype0
_
1lstm/rnn/TensorArrayUnstack/strided_slice/stack_2Const*
valueB:*
dtype0
�
)lstm/rnn/TensorArrayUnstack/strided_sliceStridedSlice!lstm/rnn/TensorArrayUnstack/Shape/lstm/rnn/TensorArrayUnstack/strided_slice/stack1lstm/rnn/TensorArrayUnstack/strided_slice/stack_11lstm/rnn/TensorArrayUnstack/strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask 
Q
'lstm/rnn/TensorArrayUnstack/range/startConst*
value	B : *
dtype0
Q
'lstm/rnn/TensorArrayUnstack/range/deltaConst*
value	B :*
dtype0
�
!lstm/rnn/TensorArrayUnstack/rangeRange'lstm/rnn/TensorArrayUnstack/range/start)lstm/rnn/TensorArrayUnstack/strided_slice'lstm/rnn/TensorArrayUnstack/range/delta*

Tidx0
�
Clstm/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3lstm/rnn/TensorArray_1!lstm/rnn/TensorArrayUnstack/rangelstm/rnn/transposelstm/rnn/TensorArray_1:1*
T0*%
_class
loc:@lstm/rnn/transpose
<
lstm/rnn/Maximum/xConst*
value	B :*
dtype0
R
lstm/rnn/MaximumMaximumlstm/rnn/Maximum/xlstm/rnn/strided_slice_1*
T0
P
lstm/rnn/MinimumMinimumlstm/rnn/strided_slice_1lstm/rnn/Maximum*
T0
J
 lstm/rnn/while/iteration_counterConst*
value	B : *
dtype0
�
lstm/rnn/while/EnterEnter lstm/rnn/while/iteration_counter*
T0*
is_constant( *
parallel_iterations *,

frame_namelstm/rnn/while/while_context
�
lstm/rnn/while/Enter_1Enterlstm/rnn/time*
T0*
is_constant( *
parallel_iterations *,

frame_namelstm/rnn/while/while_context
�
lstm/rnn/while/Enter_2Enterlstm/rnn/TensorArray:1*
T0*
is_constant( *
parallel_iterations *,

frame_namelstm/rnn/while/while_context
�
lstm/rnn/while/Enter_3Enterlstm/strided_slice*
T0*
is_constant( *
parallel_iterations *,

frame_namelstm/rnn/while/while_context
�
lstm/rnn/while/Enter_4Enterlstm/strided_slice_1*
T0*
is_constant( *
parallel_iterations *,

frame_namelstm/rnn/while/while_context
c
lstm/rnn/while/MergeMergelstm/rnn/while/Enterlstm/rnn/while/NextIteration*
T0*
N
i
lstm/rnn/while/Merge_1Mergelstm/rnn/while/Enter_1lstm/rnn/while/NextIteration_1*
T0*
N
i
lstm/rnn/while/Merge_2Mergelstm/rnn/while/Enter_2lstm/rnn/while/NextIteration_2*
T0*
N
i
lstm/rnn/while/Merge_3Mergelstm/rnn/while/Enter_3lstm/rnn/while/NextIteration_3*
T0*
N
i
lstm/rnn/while/Merge_4Mergelstm/rnn/while/Enter_4lstm/rnn/while/NextIteration_4*
T0*
N
U
lstm/rnn/while/LessLesslstm/rnn/while/Mergelstm/rnn/while/Less/Enter*
T0
�
lstm/rnn/while/Less/EnterEnterlstm/rnn/strided_slice_1*
T0*
is_constant(*
parallel_iterations *,

frame_namelstm/rnn/while/while_context
[
lstm/rnn/while/Less_1Lesslstm/rnn/while/Merge_1lstm/rnn/while/Less_1/Enter*
T0
�
lstm/rnn/while/Less_1/EnterEnterlstm/rnn/Minimum*
T0*
is_constant(*
parallel_iterations *,

frame_namelstm/rnn/while/while_context
S
lstm/rnn/while/LogicalAnd
LogicalAndlstm/rnn/while/Lesslstm/rnn/while/Less_1
>
lstm/rnn/while/LoopCondLoopCondlstm/rnn/while/LogicalAnd
�
lstm/rnn/while/SwitchSwitchlstm/rnn/while/Mergelstm/rnn/while/LoopCond*
T0*'
_class
loc:@lstm/rnn/while/Merge
�
lstm/rnn/while/Switch_1Switchlstm/rnn/while/Merge_1lstm/rnn/while/LoopCond*
T0*)
_class
loc:@lstm/rnn/while/Merge_1
�
lstm/rnn/while/Switch_2Switchlstm/rnn/while/Merge_2lstm/rnn/while/LoopCond*
T0*)
_class
loc:@lstm/rnn/while/Merge_2
�
lstm/rnn/while/Switch_3Switchlstm/rnn/while/Merge_3lstm/rnn/while/LoopCond*
T0*)
_class
loc:@lstm/rnn/while/Merge_3
�
lstm/rnn/while/Switch_4Switchlstm/rnn/while/Merge_4lstm/rnn/while/LoopCond*
T0*)
_class
loc:@lstm/rnn/while/Merge_4
E
lstm/rnn/while/IdentityIdentitylstm/rnn/while/Switch:1*
T0
I
lstm/rnn/while/Identity_1Identitylstm/rnn/while/Switch_1:1*
T0
I
lstm/rnn/while/Identity_2Identitylstm/rnn/while/Switch_2:1*
T0
I
lstm/rnn/while/Identity_3Identitylstm/rnn/while/Switch_3:1*
T0
I
lstm/rnn/while/Identity_4Identitylstm/rnn/while/Switch_4:1*
T0
X
lstm/rnn/while/add/yConst^lstm/rnn/while/Identity*
value	B :*
dtype0
S
lstm/rnn/while/addAddV2lstm/rnn/while/Identitylstm/rnn/while/add/y*
T0
�
 lstm/rnn/while/TensorArrayReadV3TensorArrayReadV3&lstm/rnn/while/TensorArrayReadV3/Enterlstm/rnn/while/Identity_1(lstm/rnn/while/TensorArrayReadV3/Enter_1*
dtype0
�
&lstm/rnn/while/TensorArrayReadV3/EnterEnterlstm/rnn/TensorArray_1*
T0*
is_constant(*
parallel_iterations *,

frame_namelstm/rnn/while/while_context
�
(lstm/rnn/while/TensorArrayReadV3/Enter_1EnterClstm/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
T0*
is_constant(*
parallel_iterations *,

frame_namelstm/rnn/while/while_context
�
@lstm/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/shapeConst*2
_class(
&$loc:@lstm/rnn/basic_lstm_cell/kernel*
valueB"�      *
dtype0
�
>lstm/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/minConst*2
_class(
&$loc:@lstm/rnn/basic_lstm_cell/kernel*
valueB
 *����*
dtype0
�
>lstm/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/maxConst*2
_class(
&$loc:@lstm/rnn/basic_lstm_cell/kernel*
valueB
 *���=*
dtype0
�
Hlstm/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/RandomUniformRandomUniform@lstm/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/shape*
seed�8*
T0*2
_class(
&$loc:@lstm/rnn/basic_lstm_cell/kernel*
dtype0*
seed2�
�
>lstm/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/subSub>lstm/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/max>lstm/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/min*
T0*2
_class(
&$loc:@lstm/rnn/basic_lstm_cell/kernel
�
>lstm/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/mulMulHlstm/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/RandomUniform>lstm/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/sub*
T0*2
_class(
&$loc:@lstm/rnn/basic_lstm_cell/kernel
�
:lstm/rnn/basic_lstm_cell/kernel/Initializer/random_uniformAdd>lstm/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/mul>lstm/rnn/basic_lstm_cell/kernel/Initializer/random_uniform/min*
T0*2
_class(
&$loc:@lstm/rnn/basic_lstm_cell/kernel
�
lstm/rnn/basic_lstm_cell/kernel
VariableV2*
shape:
��*
shared_name *2
_class(
&$loc:@lstm/rnn/basic_lstm_cell/kernel*
dtype0*
	container 
�
&lstm/rnn/basic_lstm_cell/kernel/AssignAssignlstm/rnn/basic_lstm_cell/kernel:lstm/rnn/basic_lstm_cell/kernel/Initializer/random_uniform*
use_locking(*
T0*2
_class(
&$loc:@lstm/rnn/basic_lstm_cell/kernel*
validate_shape(
Z
$lstm/rnn/basic_lstm_cell/kernel/readIdentitylstm/rnn/basic_lstm_cell/kernel*
T0
�
/lstm/rnn/basic_lstm_cell/bias/Initializer/zerosConst*0
_class&
$"loc:@lstm/rnn/basic_lstm_cell/bias*
valueB�*    *
dtype0
�
lstm/rnn/basic_lstm_cell/bias
VariableV2*
shape:�*
shared_name *0
_class&
$"loc:@lstm/rnn/basic_lstm_cell/bias*
dtype0*
	container 
�
$lstm/rnn/basic_lstm_cell/bias/AssignAssignlstm/rnn/basic_lstm_cell/bias/lstm/rnn/basic_lstm_cell/bias/Initializer/zeros*
use_locking(*
T0*0
_class&
$"loc:@lstm/rnn/basic_lstm_cell/bias*
validate_shape(
V
"lstm/rnn/basic_lstm_cell/bias/readIdentitylstm/rnn/basic_lstm_cell/bias*
T0
h
$lstm/rnn/while/basic_lstm_cell/ConstConst^lstm/rnn/while/Identity*
value	B :*
dtype0
n
*lstm/rnn/while/basic_lstm_cell/concat/axisConst^lstm/rnn/while/Identity*
value	B :*
dtype0
�
%lstm/rnn/while/basic_lstm_cell/concatConcatV2 lstm/rnn/while/TensorArrayReadV3lstm/rnn/while/Identity_4*lstm/rnn/while/basic_lstm_cell/concat/axis*

Tidx0*
T0*
N
�
%lstm/rnn/while/basic_lstm_cell/MatMulMatMul%lstm/rnn/while/basic_lstm_cell/concat+lstm/rnn/while/basic_lstm_cell/MatMul/Enter*
transpose_b( *
T0*
transpose_a( 
�
+lstm/rnn/while/basic_lstm_cell/MatMul/EnterEnter$lstm/rnn/basic_lstm_cell/kernel/read*
T0*
is_constant(*
parallel_iterations *,

frame_namelstm/rnn/while/while_context
�
&lstm/rnn/while/basic_lstm_cell/BiasAddBiasAdd%lstm/rnn/while/basic_lstm_cell/MatMul,lstm/rnn/while/basic_lstm_cell/BiasAdd/Enter*
T0*
data_formatNHWC
�
,lstm/rnn/while/basic_lstm_cell/BiasAdd/EnterEnter"lstm/rnn/basic_lstm_cell/bias/read*
T0*
is_constant(*
parallel_iterations *,

frame_namelstm/rnn/while/while_context
j
&lstm/rnn/while/basic_lstm_cell/Const_1Const^lstm/rnn/while/Identity*
value	B :*
dtype0
�
$lstm/rnn/while/basic_lstm_cell/splitSplit$lstm/rnn/while/basic_lstm_cell/Const&lstm/rnn/while/basic_lstm_cell/BiasAdd*
T0*
	num_split
m
&lstm/rnn/while/basic_lstm_cell/Const_2Const^lstm/rnn/while/Identity*
valueB
 *  �?*
dtype0
�
"lstm/rnn/while/basic_lstm_cell/AddAdd&lstm/rnn/while/basic_lstm_cell/split:2&lstm/rnn/while/basic_lstm_cell/Const_2*
T0
^
&lstm/rnn/while/basic_lstm_cell/SigmoidSigmoid"lstm/rnn/while/basic_lstm_cell/Add*
T0
u
"lstm/rnn/while/basic_lstm_cell/MulMullstm/rnn/while/Identity_3&lstm/rnn/while/basic_lstm_cell/Sigmoid*
T0
b
(lstm/rnn/while/basic_lstm_cell/Sigmoid_1Sigmoid$lstm/rnn/while/basic_lstm_cell/split*
T0
\
#lstm/rnn/while/basic_lstm_cell/TanhTanh&lstm/rnn/while/basic_lstm_cell/split:1*
T0
�
$lstm/rnn/while/basic_lstm_cell/Mul_1Mul(lstm/rnn/while/basic_lstm_cell/Sigmoid_1#lstm/rnn/while/basic_lstm_cell/Tanh*
T0
~
$lstm/rnn/while/basic_lstm_cell/Add_1Add"lstm/rnn/while/basic_lstm_cell/Mul$lstm/rnn/while/basic_lstm_cell/Mul_1*
T0
\
%lstm/rnn/while/basic_lstm_cell/Tanh_1Tanh$lstm/rnn/while/basic_lstm_cell/Add_1*
T0
d
(lstm/rnn/while/basic_lstm_cell/Sigmoid_2Sigmoid&lstm/rnn/while/basic_lstm_cell/split:3*
T0
�
$lstm/rnn/while/basic_lstm_cell/Mul_2Mul%lstm/rnn/while/basic_lstm_cell/Tanh_1(lstm/rnn/while/basic_lstm_cell/Sigmoid_2*
T0
�
2lstm/rnn/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV38lstm/rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enterlstm/rnn/while/Identity_1$lstm/rnn/while/basic_lstm_cell/Mul_2lstm/rnn/while/Identity_2*
T0*7
_class-
+)loc:@lstm/rnn/while/basic_lstm_cell/Mul_2
�
8lstm/rnn/while/TensorArrayWrite/TensorArrayWriteV3/EnterEnterlstm/rnn/TensorArray*
T0*7
_class-
+)loc:@lstm/rnn/while/basic_lstm_cell/Mul_2*
is_constant(*
parallel_iterations *,

frame_namelstm/rnn/while/while_context
Z
lstm/rnn/while/add_1/yConst^lstm/rnn/while/Identity*
value	B :*
dtype0
Y
lstm/rnn/while/add_1AddV2lstm/rnn/while/Identity_1lstm/rnn/while/add_1/y*
T0
J
lstm/rnn/while/NextIterationNextIterationlstm/rnn/while/add*
T0
N
lstm/rnn/while/NextIteration_1NextIterationlstm/rnn/while/add_1*
T0
l
lstm/rnn/while/NextIteration_2NextIteration2lstm/rnn/while/TensorArrayWrite/TensorArrayWriteV3*
T0
^
lstm/rnn/while/NextIteration_3NextIteration$lstm/rnn/while/basic_lstm_cell/Add_1*
T0
^
lstm/rnn/while/NextIteration_4NextIteration$lstm/rnn/while/basic_lstm_cell/Mul_2*
T0
;
lstm/rnn/while/ExitExitlstm/rnn/while/Switch*
T0
?
lstm/rnn/while/Exit_1Exitlstm/rnn/while/Switch_1*
T0
?
lstm/rnn/while/Exit_2Exitlstm/rnn/while/Switch_2*
T0
?
lstm/rnn/while/Exit_3Exitlstm/rnn/while/Switch_3*
T0
?
lstm/rnn/while/Exit_4Exitlstm/rnn/while/Switch_4*
T0
�
+lstm/rnn/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3lstm/rnn/TensorArraylstm/rnn/while/Exit_2*'
_class
loc:@lstm/rnn/TensorArray
x
%lstm/rnn/TensorArrayStack/range/startConst*'
_class
loc:@lstm/rnn/TensorArray*
value	B : *
dtype0
x
%lstm/rnn/TensorArrayStack/range/deltaConst*'
_class
loc:@lstm/rnn/TensorArray*
value	B :*
dtype0
�
lstm/rnn/TensorArrayStack/rangeRange%lstm/rnn/TensorArrayStack/range/start+lstm/rnn/TensorArrayStack/TensorArraySizeV3%lstm/rnn/TensorArrayStack/range/delta*

Tidx0*'
_class
loc:@lstm/rnn/TensorArray
�
-lstm/rnn/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3lstm/rnn/TensorArraylstm/rnn/TensorArrayStack/rangelstm/rnn/while/Exit_2*%
element_shape:����������*'
_class
loc:@lstm/rnn/TensorArray*
dtype0
?
lstm/rnn/Const_1Const*
valueB:�*
dtype0
9
lstm/rnn/Rank_1Const*
value	B :*
dtype0
@
lstm/rnn/range_1/startConst*
value	B :*
dtype0
@
lstm/rnn/range_1/deltaConst*
value	B :*
dtype0
f
lstm/rnn/range_1Rangelstm/rnn/range_1/startlstm/rnn/Rank_1lstm/rnn/range_1/delta*

Tidx0
O
lstm/rnn/concat_2/values_0Const*
valueB"       *
dtype0
@
lstm/rnn/concat_2/axisConst*
value	B : *
dtype0
�
lstm/rnn/concat_2ConcatV2lstm/rnn/concat_2/values_0lstm/rnn/range_1lstm/rnn/concat_2/axis*

Tidx0*
T0*
N
y
lstm/rnn/transpose_1	Transpose-lstm/rnn/TensorArrayStack/TensorArrayGatherV3lstm/rnn/concat_2*
Tperm0*
T0
D
Reshape_2/shapeConst*
valueB"�����   *
dtype0
R
	Reshape_2Reshapelstm/rnn/transpose_1Reshape_2/shape*
T0*
Tshape0
7
concat_3/axisConst*
value	B :*
dtype0
o
concat_3ConcatV2lstm/rnn/while/Exit_3lstm/rnn/while/Exit_4concat_3/axis*

Tidx0*
T0*
N
,
recurrent_outIdentityconcat_3*
T0
�
/dense/kernel/Initializer/truncated_normal/shapeConst*
_class
loc:@dense/kernel*
valueB"�      *
dtype0
|
.dense/kernel/Initializer/truncated_normal/meanConst*
_class
loc:@dense/kernel*
valueB
 *    *
dtype0
~
0dense/kernel/Initializer/truncated_normal/stddevConst*
_class
loc:@dense/kernel*
valueB
 *s%<*
dtype0
�
9dense/kernel/Initializer/truncated_normal/TruncatedNormalTruncatedNormal/dense/kernel/Initializer/truncated_normal/shape*
seed�8*
T0*
_class
loc:@dense/kernel*
dtype0*
seed2�
�
-dense/kernel/Initializer/truncated_normal/mulMul9dense/kernel/Initializer/truncated_normal/TruncatedNormal0dense/kernel/Initializer/truncated_normal/stddev*
T0*
_class
loc:@dense/kernel
�
)dense/kernel/Initializer/truncated_normalAdd-dense/kernel/Initializer/truncated_normal/mul.dense/kernel/Initializer/truncated_normal/mean*
T0*
_class
loc:@dense/kernel
�
dense/kernel
VariableV2*
shape:	�*
shared_name *
_class
loc:@dense/kernel*
dtype0*
	container 
�
dense/kernel/AssignAssigndense/kernel)dense/kernel/Initializer/truncated_normal*
use_locking(*
T0*
_class
loc:@dense/kernel*
validate_shape(
U
dense/kernel/readIdentitydense/kernel*
T0*
_class
loc:@dense/kernel
c
dense/MatMulMatMul	Reshape_2dense/kernel/read*
transpose_b( *
T0*
transpose_a( 
A
action_probs/concat_dimConst*
value	B :*
dtype0
<
action_probs/action_probsIdentitydense/MatMul*
T0
F
action_masksPlaceholder*
shape:���������*
dtype0
J
strided_slice_2/stackConst*
valueB"        *
dtype0
L
strided_slice_2/stack_1Const*
valueB"       *
dtype0
L
strided_slice_2/stack_2Const*
valueB"      *
dtype0
�
strided_slice_2StridedSliceaction_probs/action_probsstrided_slice_2/stackstrided_slice_2/stack_1strided_slice_2/stack_2*
T0*
Index0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask
J
strided_slice_3/stackConst*
valueB"        *
dtype0
L
strided_slice_3/stack_1Const*
valueB"       *
dtype0
L
strided_slice_3/stack_2Const*
valueB"      *
dtype0
�
strided_slice_3StridedSliceaction_masksstrided_slice_3/stackstrided_slice_3/stack_1strided_slice_3/stack_2*
T0*
Index0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask
,
SoftmaxSoftmaxstrided_slice_2*
T0
4
add_1/yConst*
valueB
 *���3*
dtype0
)
add_1AddV2Softmaxadd_1/y*
T0
+
MulMuladd_1strided_slice_3*
T0
?
Sum/reduction_indicesConst*
value	B :*
dtype0
L
SumSumMulSum/reduction_indices*

Tidx0*
	keep_dims(*
T0
%
truedivRealDivMulSum*
T0
4
add_2/yConst*
valueB
 *���3*
dtype0
)
add_2AddV2truedivadd_2/y*
T0

LogLogadd_2*
T0
M
#multinomial/Multinomial/num_samplesConst*
value	B :*
dtype0
�
multinomial/MultinomialMultinomialLog#multinomial/Multinomial/num_samples*
seed�8*
output_dtype0	*
T0*
seed2�
=
concat_4/concat_dimConst*
value	B :*
dtype0
=
concat_4/concatIdentitymultinomial/Multinomial*
T0	
=
concat_5/concat_dimConst*
value	B :*
dtype0
-
concat_5/concatIdentitytruediv*
T0
4
add_3/yConst*
valueB
 *���3*
dtype0
)
add_3AddV2truedivadd_3/y*
T0

Log_1Logadd_3*
T0
=
concat_6/concat_dimConst*
value	B :*
dtype0
+
concat_6/concatIdentityLog_1*
T0
.
IdentityIdentityconcat_4/concat*
T0	
,
actionIdentityconcat_6/concat*
T0
�
7extrinsic_value/kernel/Initializer/random_uniform/shapeConst*)
_class
loc:@extrinsic_value/kernel*
valueB"�      *
dtype0
�
5extrinsic_value/kernel/Initializer/random_uniform/minConst*)
_class
loc:@extrinsic_value/kernel*
valueB
 *n�\�*
dtype0
�
5extrinsic_value/kernel/Initializer/random_uniform/maxConst*)
_class
loc:@extrinsic_value/kernel*
valueB
 *n�\>*
dtype0
�
?extrinsic_value/kernel/Initializer/random_uniform/RandomUniformRandomUniform7extrinsic_value/kernel/Initializer/random_uniform/shape*
seed�8*
T0*)
_class
loc:@extrinsic_value/kernel*
dtype0*
seed2�
�
5extrinsic_value/kernel/Initializer/random_uniform/subSub5extrinsic_value/kernel/Initializer/random_uniform/max5extrinsic_value/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@extrinsic_value/kernel
�
5extrinsic_value/kernel/Initializer/random_uniform/mulMul?extrinsic_value/kernel/Initializer/random_uniform/RandomUniform5extrinsic_value/kernel/Initializer/random_uniform/sub*
T0*)
_class
loc:@extrinsic_value/kernel
�
1extrinsic_value/kernel/Initializer/random_uniformAdd5extrinsic_value/kernel/Initializer/random_uniform/mul5extrinsic_value/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@extrinsic_value/kernel
�
extrinsic_value/kernel
VariableV2*
shape:	�*
shared_name *)
_class
loc:@extrinsic_value/kernel*
dtype0*
	container 
�
extrinsic_value/kernel/AssignAssignextrinsic_value/kernel1extrinsic_value/kernel/Initializer/random_uniform*
use_locking(*
T0*)
_class
loc:@extrinsic_value/kernel*
validate_shape(
s
extrinsic_value/kernel/readIdentityextrinsic_value/kernel*
T0*)
_class
loc:@extrinsic_value/kernel
�
&extrinsic_value/bias/Initializer/zerosConst*'
_class
loc:@extrinsic_value/bias*
valueB*    *
dtype0
�
extrinsic_value/bias
VariableV2*
shape:*
shared_name *'
_class
loc:@extrinsic_value/bias*
dtype0*
	container 
�
extrinsic_value/bias/AssignAssignextrinsic_value/bias&extrinsic_value/bias/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@extrinsic_value/bias*
validate_shape(
m
extrinsic_value/bias/readIdentityextrinsic_value/bias*
T0*'
_class
loc:@extrinsic_value/bias
w
extrinsic_value/MatMulMatMul	Reshape_2extrinsic_value/kernel/read*
transpose_b( *
T0*
transpose_a( 
u
extrinsic_value/BiasAddBiasAddextrinsic_value/MatMulextrinsic_value/bias/read*
T0*
data_formatNHWC
I

Mean/inputPackextrinsic_value/BiasAdd*
T0*

axis *
N
@
Mean/reduction_indicesConst*
value	B : *
dtype0
V
MeanMean
Mean/inputMean/reduction_indices*

Tidx0*
	keep_dims( *
T0
G
action_holderPlaceholder*
shape:���������*
dtype0
J
strided_slice_4/stackConst*
valueB"        *
dtype0
L
strided_slice_4/stack_1Const*
valueB"       *
dtype0
L
strided_slice_4/stack_2Const*
valueB"      *
dtype0
�
strided_slice_4StridedSliceaction_holderstrided_slice_4/stackstrided_slice_4/stack_1strided_slice_4/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask
?
one_hot_1/on_valueConst*
valueB
 *  �?*
dtype0
@
one_hot_1/off_valueConst*
valueB
 *    *
dtype0
9
one_hot_1/depthConst*
value	B :*
dtype0
�
	one_hot_1OneHotstrided_slice_4one_hot_1/depthone_hot_1/on_valueone_hot_1/off_value*
T0*
TI0*
axis���������
=
concat_7/concat_dimConst*
value	B :*
dtype0
/
concat_7/concatIdentity	one_hot_1*
T0
6
StopGradientStopGradientconcat_7/concat*
T0
K
old_probabilitiesPlaceholder*
shape:���������*
dtype0
J
strided_slice_5/stackConst*
valueB"        *
dtype0
L
strided_slice_5/stack_1Const*
valueB"       *
dtype0
L
strided_slice_5/stack_2Const*
valueB"      *
dtype0
�
strided_slice_5StridedSliceold_probabilitiesstrided_slice_5/stackstrided_slice_5/stack_1strided_slice_5/stack_2*
T0*
Index0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask
J
strided_slice_6/stackConst*
valueB"        *
dtype0
L
strided_slice_6/stack_1Const*
valueB"       *
dtype0
L
strided_slice_6/stack_2Const*
valueB"      *
dtype0
�
strided_slice_6StridedSliceaction_masksstrided_slice_6/stackstrided_slice_6/stack_1strided_slice_6/stack_2*
T0*
Index0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask
.
	Softmax_1Softmaxstrided_slice_5*
T0
4
add_4/yConst*
valueB
 *���3*
dtype0
+
add_4AddV2	Softmax_1add_4/y*
T0
-
Mul_1Muladd_4strided_slice_6*
T0
A
Sum_1/reduction_indicesConst*
value	B :*
dtype0
R
Sum_1SumMul_1Sum_1/reduction_indices*

Tidx0*
	keep_dims(*
T0
+
	truediv_1RealDivMul_1Sum_1*
T0
4
add_5/yConst*
valueB
 *���3*
dtype0
+
add_5AddV2	truediv_1add_5/y*
T0

Log_2Logadd_5*
T0
O
%multinomial_1/Multinomial/num_samplesConst*
value	B :*
dtype0
�
multinomial_1/MultinomialMultinomialLog_2%multinomial_1/Multinomial/num_samples*
seed�8*
output_dtype0	*
T0*
seed2�
=
concat_8/concat_dimConst*
value	B :*
dtype0
?
concat_8/concatIdentitymultinomial_1/Multinomial*
T0	
=
concat_9/concat_dimConst*
value	B :*
dtype0
/
concat_9/concatIdentity	truediv_1*
T0
4
add_6/yConst*
valueB
 *���3*
dtype0
+
add_6AddV2	truediv_1add_6/y*
T0

Log_3Logadd_6*
T0
>
concat_10/concat_dimConst*
value	B :*
dtype0
,
concat_10/concatIdentityLog_3*
T0
J
strided_slice_7/stackConst*
valueB"        *
dtype0
L
strided_slice_7/stack_1Const*
valueB"       *
dtype0
L
strided_slice_7/stack_2Const*
valueB"      *
dtype0
�
strided_slice_7StridedSliceaction_probs/action_probsstrided_slice_7/stackstrided_slice_7/stack_1strided_slice_7/stack_2*
T0*
Index0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask
.
	Softmax_2Softmaxstrided_slice_7*
T0
J
strided_slice_8/stackConst*
valueB"        *
dtype0
L
strided_slice_8/stack_1Const*
valueB"       *
dtype0
L
strided_slice_8/stack_2Const*
valueB"      *
dtype0
�
strided_slice_8StridedSliceaction_probs/action_probsstrided_slice_8/stackstrided_slice_8/stack_1strided_slice_8/stack_2*
T0*
Index0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask
P
&softmax_cross_entropy_with_logits/RankConst*
value	B :*
dtype0
Z
'softmax_cross_entropy_with_logits/ShapeShapestrided_slice_8*
T0*
out_type0
R
(softmax_cross_entropy_with_logits/Rank_1Const*
value	B :*
dtype0
\
)softmax_cross_entropy_with_logits/Shape_1Shapestrided_slice_8*
T0*
out_type0
Q
'softmax_cross_entropy_with_logits/Sub/yConst*
value	B :*
dtype0
�
%softmax_cross_entropy_with_logits/SubSub(softmax_cross_entropy_with_logits/Rank_1'softmax_cross_entropy_with_logits/Sub/y*
T0
z
-softmax_cross_entropy_with_logits/Slice/beginPack%softmax_cross_entropy_with_logits/Sub*
T0*

axis *
N
Z
,softmax_cross_entropy_with_logits/Slice/sizeConst*
valueB:*
dtype0
�
'softmax_cross_entropy_with_logits/SliceSlice)softmax_cross_entropy_with_logits/Shape_1-softmax_cross_entropy_with_logits/Slice/begin,softmax_cross_entropy_with_logits/Slice/size*
T0*
Index0
h
1softmax_cross_entropy_with_logits/concat/values_0Const*
valueB:
���������*
dtype0
W
-softmax_cross_entropy_with_logits/concat/axisConst*
value	B : *
dtype0
�
(softmax_cross_entropy_with_logits/concatConcatV21softmax_cross_entropy_with_logits/concat/values_0'softmax_cross_entropy_with_logits/Slice-softmax_cross_entropy_with_logits/concat/axis*

Tidx0*
T0*
N
�
)softmax_cross_entropy_with_logits/ReshapeReshapestrided_slice_8(softmax_cross_entropy_with_logits/concat*
T0*
Tshape0
R
(softmax_cross_entropy_with_logits/Rank_2Const*
value	B :*
dtype0
V
)softmax_cross_entropy_with_logits/Shape_2Shape	Softmax_2*
T0*
out_type0
S
)softmax_cross_entropy_with_logits/Sub_1/yConst*
value	B :*
dtype0
�
'softmax_cross_entropy_with_logits/Sub_1Sub(softmax_cross_entropy_with_logits/Rank_2)softmax_cross_entropy_with_logits/Sub_1/y*
T0
~
/softmax_cross_entropy_with_logits/Slice_1/beginPack'softmax_cross_entropy_with_logits/Sub_1*
T0*

axis *
N
\
.softmax_cross_entropy_with_logits/Slice_1/sizeConst*
valueB:*
dtype0
�
)softmax_cross_entropy_with_logits/Slice_1Slice)softmax_cross_entropy_with_logits/Shape_2/softmax_cross_entropy_with_logits/Slice_1/begin.softmax_cross_entropy_with_logits/Slice_1/size*
T0*
Index0
j
3softmax_cross_entropy_with_logits/concat_1/values_0Const*
valueB:
���������*
dtype0
Y
/softmax_cross_entropy_with_logits/concat_1/axisConst*
value	B : *
dtype0
�
*softmax_cross_entropy_with_logits/concat_1ConcatV23softmax_cross_entropy_with_logits/concat_1/values_0)softmax_cross_entropy_with_logits/Slice_1/softmax_cross_entropy_with_logits/concat_1/axis*

Tidx0*
T0*
N
�
+softmax_cross_entropy_with_logits/Reshape_1Reshape	Softmax_2*softmax_cross_entropy_with_logits/concat_1*
T0*
Tshape0
�
!softmax_cross_entropy_with_logitsSoftmaxCrossEntropyWithLogits)softmax_cross_entropy_with_logits/Reshape+softmax_cross_entropy_with_logits/Reshape_1*
T0
S
)softmax_cross_entropy_with_logits/Sub_2/yConst*
value	B :*
dtype0
�
'softmax_cross_entropy_with_logits/Sub_2Sub&softmax_cross_entropy_with_logits/Rank)softmax_cross_entropy_with_logits/Sub_2/y*
T0
]
/softmax_cross_entropy_with_logits/Slice_2/beginConst*
valueB: *
dtype0
}
.softmax_cross_entropy_with_logits/Slice_2/sizePack'softmax_cross_entropy_with_logits/Sub_2*
T0*

axis *
N
�
)softmax_cross_entropy_with_logits/Slice_2Slice'softmax_cross_entropy_with_logits/Shape/softmax_cross_entropy_with_logits/Slice_2/begin.softmax_cross_entropy_with_logits/Slice_2/size*
T0*
Index0
�
+softmax_cross_entropy_with_logits/Reshape_2Reshape!softmax_cross_entropy_with_logits)softmax_cross_entropy_with_logits/Slice_2*
T0*
Tshape0
X
stackPack+softmax_cross_entropy_with_logits/Reshape_2*
T0*

axis*
N
A
Sum_2/reduction_indicesConst*
value	B :*
dtype0
R
Sum_2SumstackSum_2/reduction_indices*

Tidx0*
	keep_dims( *
T0
J
strided_slice_9/stackConst*
valueB"        *
dtype0
L
strided_slice_9/stack_1Const*
valueB"       *
dtype0
L
strided_slice_9/stack_2Const*
valueB"      *
dtype0
�
strided_slice_9StridedSliceconcat_7/concatstrided_slice_9/stackstrided_slice_9/stack_1strided_slice_9/stack_2*
T0*
Index0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask
K
strided_slice_10/stackConst*
valueB"        *
dtype0
M
strided_slice_10/stack_1Const*
valueB"       *
dtype0
M
strided_slice_10/stack_2Const*
valueB"      *
dtype0
�
strided_slice_10StridedSliceconcat_6/concatstrided_slice_10/stackstrided_slice_10/stack_1strided_slice_10/stack_2*
T0*
Index0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask
R
(softmax_cross_entropy_with_logits_1/RankConst*
value	B :*
dtype0
]
)softmax_cross_entropy_with_logits_1/ShapeShapestrided_slice_10*
T0*
out_type0
T
*softmax_cross_entropy_with_logits_1/Rank_1Const*
value	B :*
dtype0
_
+softmax_cross_entropy_with_logits_1/Shape_1Shapestrided_slice_10*
T0*
out_type0
S
)softmax_cross_entropy_with_logits_1/Sub/yConst*
value	B :*
dtype0
�
'softmax_cross_entropy_with_logits_1/SubSub*softmax_cross_entropy_with_logits_1/Rank_1)softmax_cross_entropy_with_logits_1/Sub/y*
T0
~
/softmax_cross_entropy_with_logits_1/Slice/beginPack'softmax_cross_entropy_with_logits_1/Sub*
T0*

axis *
N
\
.softmax_cross_entropy_with_logits_1/Slice/sizeConst*
valueB:*
dtype0
�
)softmax_cross_entropy_with_logits_1/SliceSlice+softmax_cross_entropy_with_logits_1/Shape_1/softmax_cross_entropy_with_logits_1/Slice/begin.softmax_cross_entropy_with_logits_1/Slice/size*
T0*
Index0
j
3softmax_cross_entropy_with_logits_1/concat/values_0Const*
valueB:
���������*
dtype0
Y
/softmax_cross_entropy_with_logits_1/concat/axisConst*
value	B : *
dtype0
�
*softmax_cross_entropy_with_logits_1/concatConcatV23softmax_cross_entropy_with_logits_1/concat/values_0)softmax_cross_entropy_with_logits_1/Slice/softmax_cross_entropy_with_logits_1/concat/axis*

Tidx0*
T0*
N
�
+softmax_cross_entropy_with_logits_1/ReshapeReshapestrided_slice_10*softmax_cross_entropy_with_logits_1/concat*
T0*
Tshape0
T
*softmax_cross_entropy_with_logits_1/Rank_2Const*
value	B :*
dtype0
^
+softmax_cross_entropy_with_logits_1/Shape_2Shapestrided_slice_9*
T0*
out_type0
U
+softmax_cross_entropy_with_logits_1/Sub_1/yConst*
value	B :*
dtype0
�
)softmax_cross_entropy_with_logits_1/Sub_1Sub*softmax_cross_entropy_with_logits_1/Rank_2+softmax_cross_entropy_with_logits_1/Sub_1/y*
T0
�
1softmax_cross_entropy_with_logits_1/Slice_1/beginPack)softmax_cross_entropy_with_logits_1/Sub_1*
T0*

axis *
N
^
0softmax_cross_entropy_with_logits_1/Slice_1/sizeConst*
valueB:*
dtype0
�
+softmax_cross_entropy_with_logits_1/Slice_1Slice+softmax_cross_entropy_with_logits_1/Shape_21softmax_cross_entropy_with_logits_1/Slice_1/begin0softmax_cross_entropy_with_logits_1/Slice_1/size*
T0*
Index0
l
5softmax_cross_entropy_with_logits_1/concat_1/values_0Const*
valueB:
���������*
dtype0
[
1softmax_cross_entropy_with_logits_1/concat_1/axisConst*
value	B : *
dtype0
�
,softmax_cross_entropy_with_logits_1/concat_1ConcatV25softmax_cross_entropy_with_logits_1/concat_1/values_0+softmax_cross_entropy_with_logits_1/Slice_11softmax_cross_entropy_with_logits_1/concat_1/axis*

Tidx0*
T0*
N
�
-softmax_cross_entropy_with_logits_1/Reshape_1Reshapestrided_slice_9,softmax_cross_entropy_with_logits_1/concat_1*
T0*
Tshape0
�
#softmax_cross_entropy_with_logits_1SoftmaxCrossEntropyWithLogits+softmax_cross_entropy_with_logits_1/Reshape-softmax_cross_entropy_with_logits_1/Reshape_1*
T0
U
+softmax_cross_entropy_with_logits_1/Sub_2/yConst*
value	B :*
dtype0
�
)softmax_cross_entropy_with_logits_1/Sub_2Sub(softmax_cross_entropy_with_logits_1/Rank+softmax_cross_entropy_with_logits_1/Sub_2/y*
T0
_
1softmax_cross_entropy_with_logits_1/Slice_2/beginConst*
valueB: *
dtype0
�
0softmax_cross_entropy_with_logits_1/Slice_2/sizePack)softmax_cross_entropy_with_logits_1/Sub_2*
T0*

axis *
N
�
+softmax_cross_entropy_with_logits_1/Slice_2Slice)softmax_cross_entropy_with_logits_1/Shape1softmax_cross_entropy_with_logits_1/Slice_2/begin0softmax_cross_entropy_with_logits_1/Slice_2/size*
T0*
Index0
�
-softmax_cross_entropy_with_logits_1/Reshape_2Reshape#softmax_cross_entropy_with_logits_1+softmax_cross_entropy_with_logits_1/Slice_2*
T0*
Tshape0
B
NegNeg-softmax_cross_entropy_with_logits_1/Reshape_2*
T0
2
stack_1PackNeg*
T0*

axis*
N
A
Sum_3/reduction_indicesConst*
value	B :*
dtype0
T
Sum_3Sumstack_1Sum_3/reduction_indices*

Tidx0*
	keep_dims(*
T0
K
strided_slice_11/stackConst*
valueB"        *
dtype0
M
strided_slice_11/stack_1Const*
valueB"       *
dtype0
M
strided_slice_11/stack_2Const*
valueB"      *
dtype0
�
strided_slice_11StridedSliceconcat_7/concatstrided_slice_11/stackstrided_slice_11/stack_1strided_slice_11/stack_2*
T0*
Index0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask
K
strided_slice_12/stackConst*
valueB"        *
dtype0
M
strided_slice_12/stack_1Const*
valueB"       *
dtype0
M
strided_slice_12/stack_2Const*
valueB"      *
dtype0
�
strided_slice_12StridedSliceconcat_10/concatstrided_slice_12/stackstrided_slice_12/stack_1strided_slice_12/stack_2*
T0*
Index0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask
R
(softmax_cross_entropy_with_logits_2/RankConst*
value	B :*
dtype0
]
)softmax_cross_entropy_with_logits_2/ShapeShapestrided_slice_12*
T0*
out_type0
T
*softmax_cross_entropy_with_logits_2/Rank_1Const*
value	B :*
dtype0
_
+softmax_cross_entropy_with_logits_2/Shape_1Shapestrided_slice_12*
T0*
out_type0
S
)softmax_cross_entropy_with_logits_2/Sub/yConst*
value	B :*
dtype0
�
'softmax_cross_entropy_with_logits_2/SubSub*softmax_cross_entropy_with_logits_2/Rank_1)softmax_cross_entropy_with_logits_2/Sub/y*
T0
~
/softmax_cross_entropy_with_logits_2/Slice/beginPack'softmax_cross_entropy_with_logits_2/Sub*
T0*

axis *
N
\
.softmax_cross_entropy_with_logits_2/Slice/sizeConst*
valueB:*
dtype0
�
)softmax_cross_entropy_with_logits_2/SliceSlice+softmax_cross_entropy_with_logits_2/Shape_1/softmax_cross_entropy_with_logits_2/Slice/begin.softmax_cross_entropy_with_logits_2/Slice/size*
T0*
Index0
j
3softmax_cross_entropy_with_logits_2/concat/values_0Const*
valueB:
���������*
dtype0
Y
/softmax_cross_entropy_with_logits_2/concat/axisConst*
value	B : *
dtype0
�
*softmax_cross_entropy_with_logits_2/concatConcatV23softmax_cross_entropy_with_logits_2/concat/values_0)softmax_cross_entropy_with_logits_2/Slice/softmax_cross_entropy_with_logits_2/concat/axis*

Tidx0*
T0*
N
�
+softmax_cross_entropy_with_logits_2/ReshapeReshapestrided_slice_12*softmax_cross_entropy_with_logits_2/concat*
T0*
Tshape0
T
*softmax_cross_entropy_with_logits_2/Rank_2Const*
value	B :*
dtype0
_
+softmax_cross_entropy_with_logits_2/Shape_2Shapestrided_slice_11*
T0*
out_type0
U
+softmax_cross_entropy_with_logits_2/Sub_1/yConst*
value	B :*
dtype0
�
)softmax_cross_entropy_with_logits_2/Sub_1Sub*softmax_cross_entropy_with_logits_2/Rank_2+softmax_cross_entropy_with_logits_2/Sub_1/y*
T0
�
1softmax_cross_entropy_with_logits_2/Slice_1/beginPack)softmax_cross_entropy_with_logits_2/Sub_1*
T0*

axis *
N
^
0softmax_cross_entropy_with_logits_2/Slice_1/sizeConst*
valueB:*
dtype0
�
+softmax_cross_entropy_with_logits_2/Slice_1Slice+softmax_cross_entropy_with_logits_2/Shape_21softmax_cross_entropy_with_logits_2/Slice_1/begin0softmax_cross_entropy_with_logits_2/Slice_1/size*
T0*
Index0
l
5softmax_cross_entropy_with_logits_2/concat_1/values_0Const*
valueB:
���������*
dtype0
[
1softmax_cross_entropy_with_logits_2/concat_1/axisConst*
value	B : *
dtype0
�
,softmax_cross_entropy_with_logits_2/concat_1ConcatV25softmax_cross_entropy_with_logits_2/concat_1/values_0+softmax_cross_entropy_with_logits_2/Slice_11softmax_cross_entropy_with_logits_2/concat_1/axis*

Tidx0*
T0*
N
�
-softmax_cross_entropy_with_logits_2/Reshape_1Reshapestrided_slice_11,softmax_cross_entropy_with_logits_2/concat_1*
T0*
Tshape0
�
#softmax_cross_entropy_with_logits_2SoftmaxCrossEntropyWithLogits+softmax_cross_entropy_with_logits_2/Reshape-softmax_cross_entropy_with_logits_2/Reshape_1*
T0
U
+softmax_cross_entropy_with_logits_2/Sub_2/yConst*
value	B :*
dtype0
�
)softmax_cross_entropy_with_logits_2/Sub_2Sub(softmax_cross_entropy_with_logits_2/Rank+softmax_cross_entropy_with_logits_2/Sub_2/y*
T0
_
1softmax_cross_entropy_with_logits_2/Slice_2/beginConst*
valueB: *
dtype0
�
0softmax_cross_entropy_with_logits_2/Slice_2/sizePack)softmax_cross_entropy_with_logits_2/Sub_2*
T0*

axis *
N
�
+softmax_cross_entropy_with_logits_2/Slice_2Slice)softmax_cross_entropy_with_logits_2/Shape1softmax_cross_entropy_with_logits_2/Slice_2/begin0softmax_cross_entropy_with_logits_2/Slice_2/size*
T0*
Index0
�
-softmax_cross_entropy_with_logits_2/Reshape_2Reshape#softmax_cross_entropy_with_logits_2+softmax_cross_entropy_with_logits_2/Slice_2*
T0*
Tshape0
D
Neg_1Neg-softmax_cross_entropy_with_logits_2/Reshape_2*
T0
4
stack_2PackNeg_1*
T0*

axis*
N
A
Sum_4/reduction_indicesConst*
value	B :*
dtype0
T
Sum_4Sumstack_2Sum_4/reduction_indices*

Tidx0*
	keep_dims(*
T0
R
%PolynomialDecay/initial_learning_rateConst*
valueB
 *RI�9*
dtype0
C
PolynomialDecay/Cast/xConst*
valueB
 *���.*
dtype0
E
PolynomialDecay/Cast_1/xConst*
valueB
 *  �?*
dtype0
X
PolynomialDecay/Cast_2Castglobal_step/read*

SrcT0*
Truncate( *

DstT0
E
PolynomialDecay/Cast_3/xConst*
valueB
 * $�H*
dtype0
F
PolynomialDecay/Minimum/yConst*
valueB
 * $�H*
dtype0
^
PolynomialDecay/MinimumMinimumPolynomialDecay/Cast_2PolynomialDecay/Minimum/y*
T0
^
PolynomialDecay/truedivRealDivPolynomialDecay/MinimumPolynomialDecay/Cast_3/x*
T0
b
PolynomialDecay/subSub%PolynomialDecay/initial_learning_ratePolynomialDecay/Cast/x*
T0
D
PolynomialDecay/sub_1/xConst*
valueB
 *  �?*
dtype0
W
PolynomialDecay/sub_1SubPolynomialDecay/sub_1/xPolynomialDecay/truediv*
T0
T
PolynomialDecay/PowPowPolynomialDecay/sub_1PolynomialDecay/Cast_1/x*
T0
M
PolynomialDecay/MulMulPolynomialDecay/subPolynomialDecay/Pow*
T0
L
PolynomialDecayAddPolynomialDecay/MulPolynomialDecay/Cast/x*
T0
G
extrinsic_returnsPlaceholder*
shape:���������*
dtype0
N
extrinsic_value_estimatePlaceholder*
shape:���������*
dtype0
@

advantagesPlaceholder*
shape:���������*
dtype0
A
ExpandDims/dimConst*
valueB :
���������*
dtype0
I

ExpandDims
ExpandDims
advantagesExpandDims/dim*

Tdim0*
T0
T
'PolynomialDecay_1/initial_learning_rateConst*
valueB
 *��L>*
dtype0
E
PolynomialDecay_1/Cast/xConst*
valueB
 *���=*
dtype0
G
PolynomialDecay_1/Cast_1/xConst*
valueB
 *  �?*
dtype0
Z
PolynomialDecay_1/Cast_2Castglobal_step/read*

SrcT0*
Truncate( *

DstT0
G
PolynomialDecay_1/Cast_3/xConst*
valueB
 * $�H*
dtype0
H
PolynomialDecay_1/Minimum/yConst*
valueB
 * $�H*
dtype0
d
PolynomialDecay_1/MinimumMinimumPolynomialDecay_1/Cast_2PolynomialDecay_1/Minimum/y*
T0
d
PolynomialDecay_1/truedivRealDivPolynomialDecay_1/MinimumPolynomialDecay_1/Cast_3/x*
T0
h
PolynomialDecay_1/subSub'PolynomialDecay_1/initial_learning_ratePolynomialDecay_1/Cast/x*
T0
F
PolynomialDecay_1/sub_1/xConst*
valueB
 *  �?*
dtype0
]
PolynomialDecay_1/sub_1SubPolynomialDecay_1/sub_1/xPolynomialDecay_1/truediv*
T0
Z
PolynomialDecay_1/PowPowPolynomialDecay_1/sub_1PolynomialDecay_1/Cast_1/x*
T0
S
PolynomialDecay_1/MulMulPolynomialDecay_1/subPolynomialDecay_1/Pow*
T0
R
PolynomialDecay_1AddPolynomialDecay_1/MulPolynomialDecay_1/Cast/x*
T0
T
'PolynomialDecay_2/initial_learning_rateConst*
valueB
 *
ף;*
dtype0
E
PolynomialDecay_2/Cast/xConst*
valueB
 *��'7*
dtype0
G
PolynomialDecay_2/Cast_1/xConst*
valueB
 *  �?*
dtype0
Z
PolynomialDecay_2/Cast_2Castglobal_step/read*

SrcT0*
Truncate( *

DstT0
G
PolynomialDecay_2/Cast_3/xConst*
valueB
 * $�H*
dtype0
H
PolynomialDecay_2/Minimum/yConst*
valueB
 * $�H*
dtype0
d
PolynomialDecay_2/MinimumMinimumPolynomialDecay_2/Cast_2PolynomialDecay_2/Minimum/y*
T0
d
PolynomialDecay_2/truedivRealDivPolynomialDecay_2/MinimumPolynomialDecay_2/Cast_3/x*
T0
h
PolynomialDecay_2/subSub'PolynomialDecay_2/initial_learning_ratePolynomialDecay_2/Cast/x*
T0
F
PolynomialDecay_2/sub_1/xConst*
valueB
 *  �?*
dtype0
]
PolynomialDecay_2/sub_1SubPolynomialDecay_2/sub_1/xPolynomialDecay_2/truediv*
T0
Z
PolynomialDecay_2/PowPowPolynomialDecay_2/sub_1PolynomialDecay_2/Cast_1/x*
T0
S
PolynomialDecay_2/MulMulPolynomialDecay_2/subPolynomialDecay_2/Pow*
T0
R
PolynomialDecay_2AddPolynomialDecay_2/MulPolynomialDecay_2/Cast/x*
T0
A
Sum_5/reduction_indicesConst*
value	B :*
dtype0
d
Sum_5Sumextrinsic_value/BiasAddSum_5/reduction_indices*

Tidx0*
	keep_dims( *
T0
4
subSubSum_5extrinsic_value_estimate*
T0
(
Neg_2NegPolynomialDecay_1*
T0
A
clip_by_value/MinimumMinimumsubPolynomialDecay_1*
T0
?
clip_by_valueMaximumclip_by_value/MinimumNeg_2*
T0
@
add_7AddV2extrinsic_value_estimateclip_by_value*
T0
A
Sum_6/reduction_indicesConst*
value	B :*
dtype0
d
Sum_6Sumextrinsic_value/BiasAddSum_6/reduction_indices*

Tidx0*
	keep_dims( *
T0
I
SquaredDifferenceSquaredDifferenceextrinsic_returnsSum_6*
T0
K
SquaredDifference_1SquaredDifferenceextrinsic_returnsadd_7*
T0
C
MaximumMaximumSquaredDifferenceSquaredDifference_1*
T0
R
DynamicPartitionDynamicPartitionMaximumCast*
num_partitions*
T0
3
ConstConst*
valueB: *
dtype0
O
Mean_1MeanDynamicPartition:1Const*

Tidx0*
	keep_dims( *
T0
9
Rank/packedPackMean_1*
T0*

axis *
N
.
RankConst*
value	B :*
dtype0
5
range/startConst*
value	B : *
dtype0
5
range/deltaConst*
value	B :*
dtype0
:
rangeRangerange/startRankrange/delta*

Tidx0
:
Mean_2/inputPackMean_1*
T0*

axis *
N
I
Mean_2MeanMean_2/inputrange*

Tidx0*
	keep_dims( *
T0
#
sub_1SubSum_3Sum_4*
T0

ExpExpsub_1*
T0
&
mul_2MulExp
ExpandDims*
T0
4
sub_2/xConst*
valueB
 *  �?*
dtype0
1
sub_2Subsub_2/xPolynomialDecay_1*
T0
4
add_8/xConst*
valueB
 *  �?*
dtype0
3
add_8AddV2add_8/xPolynomialDecay_1*
T0
7
clip_by_value_1/MinimumMinimumExpadd_8*
T0
C
clip_by_value_1Maximumclip_by_value_1/Minimumsub_2*
T0
2
mul_3Mulclip_by_value_1
ExpandDims*
T0
)
MinimumMinimummul_2mul_3*
T0
T
DynamicPartition_1DynamicPartitionMinimumCast*
num_partitions*
T0
<
Const_1Const*
valueB"       *
dtype0
S
Mean_3MeanDynamicPartition_1:1Const_1*

Tidx0*
	keep_dims( *
T0

Neg_3NegMean_3*
T0

AbsAbsNeg_3*
T0
4
mul_4/xConst*
valueB
 *   ?*
dtype0
&
mul_4Mulmul_4/xMean_2*
T0
%
add_9AddV2Neg_3mul_4*
T0
R
DynamicPartition_2DynamicPartitionSum_2Cast*
num_partitions*
T0
5
Const_2Const*
valueB: *
dtype0
S
Mean_4MeanDynamicPartition_2:1Const_2*

Tidx0*
	keep_dims( *
T0
0
mul_5MulPolynomialDecay_2Mean_4*
T0
#
sub_3Subadd_9mul_5*
T0
8
gradients/ShapeConst*
valueB *
dtype0
@
gradients/grad_ys_0Const*
valueB
 *  �?*
dtype0
W
gradients/FillFillgradients/Shapegradients/grad_ys_0*
T0*

index_type0
;
gradients/f_countConst*
value	B : *
dtype0
�
gradients/f_count_1Entergradients/f_count*
T0*
is_constant( *
parallel_iterations *,

frame_namelstm/rnn/while/while_context
X
gradients/MergeMergegradients/f_count_1gradients/NextIteration*
T0*
N
M
gradients/SwitchSwitchgradients/Mergelstm/rnn/while/LoopCond*
T0
S
gradients/Add/yConst^lstm/rnn/while/Identity*
value	B :*
dtype0
B
gradients/AddAddgradients/Switch:1gradients/Add/y*
T0
�
gradients/NextIterationNextIterationgradients/Add`^gradients/lstm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2V^gradients/lstm/rnn/while/basic_lstm_cell/Add_1_grad/BroadcastGradientArgs/StackPushV2X^gradients/lstm/rnn/while/basic_lstm_cell/Add_1_grad/BroadcastGradientArgs/StackPushV2_1T^gradients/lstm/rnn/while/basic_lstm_cell/Add_grad/BroadcastGradientArgs/StackPushV2V^gradients/lstm/rnn/while/basic_lstm_cell/Add_grad/BroadcastGradientArgs/StackPushV2_1J^gradients/lstm/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/StackPushV2V^gradients/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/BroadcastGradientArgs/StackPushV2X^gradients/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/BroadcastGradientArgs/StackPushV2_1D^gradients/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/StackPushV2F^gradients/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/StackPushV2V^gradients/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/BroadcastGradientArgs/StackPushV2X^gradients/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/BroadcastGradientArgs/StackPushV2_1D^gradients/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/StackPushV2F^gradients/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/StackPushV2T^gradients/lstm/rnn/while/basic_lstm_cell/Mul_grad/BroadcastGradientArgs/StackPushV2V^gradients/lstm/rnn/while/basic_lstm_cell/Mul_grad/BroadcastGradientArgs/StackPushV2_1B^gradients/lstm/rnn/while/basic_lstm_cell/Mul_grad/Mul/StackPushV2D^gradients/lstm/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/StackPushV2H^gradients/lstm/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPushV2J^gradients/lstm/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPushV2_1*
T0
6
gradients/f_count_2Exitgradients/Switch*
T0
;
gradients/b_countConst*
value	B :*
dtype0
�
gradients/b_count_1Entergradients/f_count_2*
T0*
is_constant( *
parallel_iterations *6

frame_name(&gradients/lstm/rnn/while/while_context
\
gradients/Merge_1Mergegradients/b_count_1gradients/NextIteration_1*
T0*
N
`
gradients/GreaterEqualGreaterEqualgradients/Merge_1gradients/GreaterEqual/Enter*
T0
�
gradients/GreaterEqual/EnterEntergradients/b_count*
T0*
is_constant(*
parallel_iterations *6

frame_name(&gradients/lstm/rnn/while/while_context
7
gradients/b_count_2LoopCondgradients/GreaterEqual
M
gradients/Switch_1Switchgradients/Merge_1gradients/b_count_2*
T0
Q
gradients/SubSubgradients/Switch_1:1gradients/GreaterEqual/Enter*
T0
�
gradients/NextIteration_1NextIterationgradients/Sub[^gradients/lstm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/b_sync*
T0
8
gradients/b_count_3Exitgradients/Switch_1*
T0
8
gradients/sub_3_grad/NegNeggradients/Fill*
T0
Y
%gradients/sub_3_grad/tuple/group_depsNoOp^gradients/Fill^gradients/sub_3_grad/Neg
�
-gradients/sub_3_grad/tuple/control_dependencyIdentitygradients/Fill&^gradients/sub_3_grad/tuple/group_deps*
T0*!
_class
loc:@gradients/Fill
�
/gradients/sub_3_grad/tuple/control_dependency_1Identitygradients/sub_3_grad/Neg&^gradients/sub_3_grad/tuple/group_deps*
T0*+
_class!
loc:@gradients/sub_3_grad/Neg
]
%gradients/add_9_grad/tuple/group_depsNoOp.^gradients/sub_3_grad/tuple/control_dependency
�
-gradients/add_9_grad/tuple/control_dependencyIdentity-gradients/sub_3_grad/tuple/control_dependency&^gradients/add_9_grad/tuple/group_deps*
T0*!
_class
loc:@gradients/Fill
�
/gradients/add_9_grad/tuple/control_dependency_1Identity-gradients/sub_3_grad/tuple/control_dependency&^gradients/add_9_grad/tuple/group_deps*
T0*!
_class
loc:@gradients/Fill
a
gradients/mul_5_grad/MulMul/gradients/sub_3_grad/tuple/control_dependency_1Mean_4*
T0
n
gradients/mul_5_grad/Mul_1Mul/gradients/sub_3_grad/tuple/control_dependency_1PolynomialDecay_2*
T0
e
%gradients/mul_5_grad/tuple/group_depsNoOp^gradients/mul_5_grad/Mul^gradients/mul_5_grad/Mul_1
�
-gradients/mul_5_grad/tuple/control_dependencyIdentitygradients/mul_5_grad/Mul&^gradients/mul_5_grad/tuple/group_deps*
T0*+
_class!
loc:@gradients/mul_5_grad/Mul
�
/gradients/mul_5_grad/tuple/control_dependency_1Identitygradients/mul_5_grad/Mul_1&^gradients/mul_5_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/mul_5_grad/Mul_1
W
gradients/Neg_3_grad/NegNeg-gradients/add_9_grad/tuple/control_dependency*
T0
a
gradients/mul_4_grad/MulMul/gradients/add_9_grad/tuple/control_dependency_1Mean_2*
T0
d
gradients/mul_4_grad/Mul_1Mul/gradients/add_9_grad/tuple/control_dependency_1mul_4/x*
T0
e
%gradients/mul_4_grad/tuple/group_depsNoOp^gradients/mul_4_grad/Mul^gradients/mul_4_grad/Mul_1
�
-gradients/mul_4_grad/tuple/control_dependencyIdentitygradients/mul_4_grad/Mul&^gradients/mul_4_grad/tuple/group_deps*
T0*+
_class!
loc:@gradients/mul_4_grad/Mul
�
/gradients/mul_4_grad/tuple/control_dependency_1Identitygradients/mul_4_grad/Mul_1&^gradients/mul_4_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/mul_4_grad/Mul_1
Q
#gradients/Mean_4_grad/Reshape/shapeConst*
valueB:*
dtype0
�
gradients/Mean_4_grad/ReshapeReshape/gradients/mul_5_grad/tuple/control_dependency_1#gradients/Mean_4_grad/Reshape/shape*
T0*
Tshape0
S
gradients/Mean_4_grad/ShapeShapeDynamicPartition_2:1*
T0*
out_type0
y
gradients/Mean_4_grad/TileTilegradients/Mean_4_grad/Reshapegradients/Mean_4_grad/Shape*

Tmultiples0*
T0
U
gradients/Mean_4_grad/Shape_1ShapeDynamicPartition_2:1*
T0*
out_type0
F
gradients/Mean_4_grad/Shape_2Const*
valueB *
dtype0
I
gradients/Mean_4_grad/ConstConst*
valueB: *
dtype0
�
gradients/Mean_4_grad/ProdProdgradients/Mean_4_grad/Shape_1gradients/Mean_4_grad/Const*

Tidx0*
	keep_dims( *
T0
K
gradients/Mean_4_grad/Const_1Const*
valueB: *
dtype0
�
gradients/Mean_4_grad/Prod_1Prodgradients/Mean_4_grad/Shape_2gradients/Mean_4_grad/Const_1*

Tidx0*
	keep_dims( *
T0
I
gradients/Mean_4_grad/Maximum/yConst*
value	B :*
dtype0
p
gradients/Mean_4_grad/MaximumMaximumgradients/Mean_4_grad/Prod_1gradients/Mean_4_grad/Maximum/y*
T0
n
gradients/Mean_4_grad/floordivFloorDivgradients/Mean_4_grad/Prodgradients/Mean_4_grad/Maximum*
T0
j
gradients/Mean_4_grad/CastCastgradients/Mean_4_grad/floordiv*

SrcT0*
Truncate( *

DstT0
i
gradients/Mean_4_grad/truedivRealDivgradients/Mean_4_grad/Tilegradients/Mean_4_grad/Cast*
T0
X
#gradients/Mean_3_grad/Reshape/shapeConst*
valueB"      *
dtype0
~
gradients/Mean_3_grad/ReshapeReshapegradients/Neg_3_grad/Neg#gradients/Mean_3_grad/Reshape/shape*
T0*
Tshape0
S
gradients/Mean_3_grad/ShapeShapeDynamicPartition_1:1*
T0*
out_type0
y
gradients/Mean_3_grad/TileTilegradients/Mean_3_grad/Reshapegradients/Mean_3_grad/Shape*

Tmultiples0*
T0
U
gradients/Mean_3_grad/Shape_1ShapeDynamicPartition_1:1*
T0*
out_type0
F
gradients/Mean_3_grad/Shape_2Const*
valueB *
dtype0
I
gradients/Mean_3_grad/ConstConst*
valueB: *
dtype0
�
gradients/Mean_3_grad/ProdProdgradients/Mean_3_grad/Shape_1gradients/Mean_3_grad/Const*

Tidx0*
	keep_dims( *
T0
K
gradients/Mean_3_grad/Const_1Const*
valueB: *
dtype0
�
gradients/Mean_3_grad/Prod_1Prodgradients/Mean_3_grad/Shape_2gradients/Mean_3_grad/Const_1*

Tidx0*
	keep_dims( *
T0
I
gradients/Mean_3_grad/Maximum/yConst*
value	B :*
dtype0
p
gradients/Mean_3_grad/MaximumMaximumgradients/Mean_3_grad/Prod_1gradients/Mean_3_grad/Maximum/y*
T0
n
gradients/Mean_3_grad/floordivFloorDivgradients/Mean_3_grad/Prodgradients/Mean_3_grad/Maximum*
T0
j
gradients/Mean_3_grad/CastCastgradients/Mean_3_grad/floordiv*

SrcT0*
Truncate( *

DstT0
i
gradients/Mean_3_grad/truedivRealDivgradients/Mean_3_grad/Tilegradients/Mean_3_grad/Cast*
T0
Q
#gradients/Mean_2_grad/Reshape/shapeConst*
valueB:*
dtype0
�
gradients/Mean_2_grad/ReshapeReshape/gradients/mul_4_grad/tuple/control_dependency_1#gradients/Mean_2_grad/Reshape/shape*
T0*
Tshape0
I
gradients/Mean_2_grad/ConstConst*
valueB:*
dtype0
y
gradients/Mean_2_grad/TileTilegradients/Mean_2_grad/Reshapegradients/Mean_2_grad/Const*

Tmultiples0*
T0
J
gradients/Mean_2_grad/Const_1Const*
valueB
 *  �?*
dtype0
l
gradients/Mean_2_grad/truedivRealDivgradients/Mean_2_grad/Tilegradients/Mean_2_grad/Const_1*
T0
>
gradients/zeros_like	ZerosLikeDynamicPartition_2*
T0
O
'gradients/DynamicPartition_2_grad/ShapeShapeCast*
T0*
out_type0
U
'gradients/DynamicPartition_2_grad/ConstConst*
valueB: *
dtype0
�
&gradients/DynamicPartition_2_grad/ProdProd'gradients/DynamicPartition_2_grad/Shape'gradients/DynamicPartition_2_grad/Const*

Tidx0*
	keep_dims( *
T0
W
-gradients/DynamicPartition_2_grad/range/startConst*
value	B : *
dtype0
W
-gradients/DynamicPartition_2_grad/range/deltaConst*
value	B :*
dtype0
�
'gradients/DynamicPartition_2_grad/rangeRange-gradients/DynamicPartition_2_grad/range/start&gradients/DynamicPartition_2_grad/Prod-gradients/DynamicPartition_2_grad/range/delta*

Tidx0
�
)gradients/DynamicPartition_2_grad/ReshapeReshape'gradients/DynamicPartition_2_grad/range'gradients/DynamicPartition_2_grad/Shape*
T0*
Tshape0
�
2gradients/DynamicPartition_2_grad/DynamicPartitionDynamicPartition)gradients/DynamicPartition_2_grad/ReshapeCast*
num_partitions*
T0
�
/gradients/DynamicPartition_2_grad/DynamicStitchDynamicStitch2gradients/DynamicPartition_2_grad/DynamicPartition4gradients/DynamicPartition_2_grad/DynamicPartition:1gradients/zeros_likegradients/Mean_4_grad/truediv*
T0*
N
R
)gradients/DynamicPartition_2_grad/Shape_1ShapeSum_2*
T0*
out_type0
�
+gradients/DynamicPartition_2_grad/Reshape_1Reshape/gradients/DynamicPartition_2_grad/DynamicStitch)gradients/DynamicPartition_2_grad/Shape_1*
T0*
Tshape0
@
gradients/zeros_like_1	ZerosLikeDynamicPartition_1*
T0
O
'gradients/DynamicPartition_1_grad/ShapeShapeCast*
T0*
out_type0
U
'gradients/DynamicPartition_1_grad/ConstConst*
valueB: *
dtype0
�
&gradients/DynamicPartition_1_grad/ProdProd'gradients/DynamicPartition_1_grad/Shape'gradients/DynamicPartition_1_grad/Const*

Tidx0*
	keep_dims( *
T0
W
-gradients/DynamicPartition_1_grad/range/startConst*
value	B : *
dtype0
W
-gradients/DynamicPartition_1_grad/range/deltaConst*
value	B :*
dtype0
�
'gradients/DynamicPartition_1_grad/rangeRange-gradients/DynamicPartition_1_grad/range/start&gradients/DynamicPartition_1_grad/Prod-gradients/DynamicPartition_1_grad/range/delta*

Tidx0
�
)gradients/DynamicPartition_1_grad/ReshapeReshape'gradients/DynamicPartition_1_grad/range'gradients/DynamicPartition_1_grad/Shape*
T0*
Tshape0
�
2gradients/DynamicPartition_1_grad/DynamicPartitionDynamicPartition)gradients/DynamicPartition_1_grad/ReshapeCast*
num_partitions*
T0
�
/gradients/DynamicPartition_1_grad/DynamicStitchDynamicStitch2gradients/DynamicPartition_1_grad/DynamicPartition4gradients/DynamicPartition_1_grad/DynamicPartition:1gradients/zeros_like_1gradients/Mean_3_grad/truediv*
T0*
N
T
)gradients/DynamicPartition_1_grad/Shape_1ShapeMinimum*
T0*
out_type0
�
+gradients/DynamicPartition_1_grad/Reshape_1Reshape/gradients/DynamicPartition_1_grad/DynamicStitch)gradients/DynamicPartition_1_grad/Shape_1*
T0*
Tshape0
l
#gradients/Mean_2/input_grad/unstackUnpackgradients/Mean_2_grad/truediv*
T0*	
num*

axis 
C
gradients/Sum_2_grad/ShapeShapestack*
T0*
out_type0
r
gradients/Sum_2_grad/SizeConst*-
_class#
!loc:@gradients/Sum_2_grad/Shape*
value	B :*
dtype0
�
gradients/Sum_2_grad/addAddV2Sum_2/reduction_indicesgradients/Sum_2_grad/Size*
T0*-
_class#
!loc:@gradients/Sum_2_grad/Shape
�
gradients/Sum_2_grad/modFloorModgradients/Sum_2_grad/addgradients/Sum_2_grad/Size*
T0*-
_class#
!loc:@gradients/Sum_2_grad/Shape
t
gradients/Sum_2_grad/Shape_1Const*-
_class#
!loc:@gradients/Sum_2_grad/Shape*
valueB *
dtype0
y
 gradients/Sum_2_grad/range/startConst*-
_class#
!loc:@gradients/Sum_2_grad/Shape*
value	B : *
dtype0
y
 gradients/Sum_2_grad/range/deltaConst*-
_class#
!loc:@gradients/Sum_2_grad/Shape*
value	B :*
dtype0
�
gradients/Sum_2_grad/rangeRange gradients/Sum_2_grad/range/startgradients/Sum_2_grad/Size gradients/Sum_2_grad/range/delta*

Tidx0*-
_class#
!loc:@gradients/Sum_2_grad/Shape
x
gradients/Sum_2_grad/Fill/valueConst*-
_class#
!loc:@gradients/Sum_2_grad/Shape*
value	B :*
dtype0
�
gradients/Sum_2_grad/FillFillgradients/Sum_2_grad/Shape_1gradients/Sum_2_grad/Fill/value*
T0*-
_class#
!loc:@gradients/Sum_2_grad/Shape*

index_type0
�
"gradients/Sum_2_grad/DynamicStitchDynamicStitchgradients/Sum_2_grad/rangegradients/Sum_2_grad/modgradients/Sum_2_grad/Shapegradients/Sum_2_grad/Fill*
T0*-
_class#
!loc:@gradients/Sum_2_grad/Shape*
N
w
gradients/Sum_2_grad/Maximum/yConst*-
_class#
!loc:@gradients/Sum_2_grad/Shape*
value	B :*
dtype0
�
gradients/Sum_2_grad/MaximumMaximum"gradients/Sum_2_grad/DynamicStitchgradients/Sum_2_grad/Maximum/y*
T0*-
_class#
!loc:@gradients/Sum_2_grad/Shape
�
gradients/Sum_2_grad/floordivFloorDivgradients/Sum_2_grad/Shapegradients/Sum_2_grad/Maximum*
T0*-
_class#
!loc:@gradients/Sum_2_grad/Shape
�
gradients/Sum_2_grad/ReshapeReshape+gradients/DynamicPartition_2_grad/Reshape_1"gradients/Sum_2_grad/DynamicStitch*
T0*
Tshape0
y
gradients/Sum_2_grad/TileTilegradients/Sum_2_grad/Reshapegradients/Sum_2_grad/floordiv*

Tmultiples0*
T0
E
gradients/Minimum_grad/ShapeShapemul_2*
T0*
out_type0
G
gradients/Minimum_grad/Shape_1Shapemul_3*
T0*
out_type0
m
gradients/Minimum_grad/Shape_2Shape+gradients/DynamicPartition_1_grad/Reshape_1*
T0*
out_type0
O
"gradients/Minimum_grad/zeros/ConstConst*
valueB
 *    *
dtype0
�
gradients/Minimum_grad/zerosFillgradients/Minimum_grad/Shape_2"gradients/Minimum_grad/zeros/Const*
T0*

index_type0
D
 gradients/Minimum_grad/LessEqual	LessEqualmul_2mul_3*
T0
�
,gradients/Minimum_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Minimum_grad/Shapegradients/Minimum_grad/Shape_1*
T0
�
gradients/Minimum_grad/SelectSelect gradients/Minimum_grad/LessEqual+gradients/DynamicPartition_1_grad/Reshape_1gradients/Minimum_grad/zeros*
T0
�
gradients/Minimum_grad/SumSumgradients/Minimum_grad/Select,gradients/Minimum_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
z
gradients/Minimum_grad/ReshapeReshapegradients/Minimum_grad/Sumgradients/Minimum_grad/Shape*
T0*
Tshape0
�
gradients/Minimum_grad/Select_1Select gradients/Minimum_grad/LessEqualgradients/Minimum_grad/zeros+gradients/DynamicPartition_1_grad/Reshape_1*
T0
�
gradients/Minimum_grad/Sum_1Sumgradients/Minimum_grad/Select_1.gradients/Minimum_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
�
 gradients/Minimum_grad/Reshape_1Reshapegradients/Minimum_grad/Sum_1gradients/Minimum_grad/Shape_1*
T0*
Tshape0
s
'gradients/Minimum_grad/tuple/group_depsNoOp^gradients/Minimum_grad/Reshape!^gradients/Minimum_grad/Reshape_1
�
/gradients/Minimum_grad/tuple/control_dependencyIdentitygradients/Minimum_grad/Reshape(^gradients/Minimum_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/Minimum_grad/Reshape
�
1gradients/Minimum_grad/tuple/control_dependency_1Identity gradients/Minimum_grad/Reshape_1(^gradients/Minimum_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients/Minimum_grad/Reshape_1
Q
#gradients/Mean_1_grad/Reshape/shapeConst*
valueB:*
dtype0
�
gradients/Mean_1_grad/ReshapeReshape#gradients/Mean_2/input_grad/unstack#gradients/Mean_1_grad/Reshape/shape*
T0*
Tshape0
Q
gradients/Mean_1_grad/ShapeShapeDynamicPartition:1*
T0*
out_type0
y
gradients/Mean_1_grad/TileTilegradients/Mean_1_grad/Reshapegradients/Mean_1_grad/Shape*

Tmultiples0*
T0
S
gradients/Mean_1_grad/Shape_1ShapeDynamicPartition:1*
T0*
out_type0
F
gradients/Mean_1_grad/Shape_2Const*
valueB *
dtype0
I
gradients/Mean_1_grad/ConstConst*
valueB: *
dtype0
�
gradients/Mean_1_grad/ProdProdgradients/Mean_1_grad/Shape_1gradients/Mean_1_grad/Const*

Tidx0*
	keep_dims( *
T0
K
gradients/Mean_1_grad/Const_1Const*
valueB: *
dtype0
�
gradients/Mean_1_grad/Prod_1Prodgradients/Mean_1_grad/Shape_2gradients/Mean_1_grad/Const_1*

Tidx0*
	keep_dims( *
T0
I
gradients/Mean_1_grad/Maximum/yConst*
value	B :*
dtype0
p
gradients/Mean_1_grad/MaximumMaximumgradients/Mean_1_grad/Prod_1gradients/Mean_1_grad/Maximum/y*
T0
n
gradients/Mean_1_grad/floordivFloorDivgradients/Mean_1_grad/Prodgradients/Mean_1_grad/Maximum*
T0
j
gradients/Mean_1_grad/CastCastgradients/Mean_1_grad/floordiv*

SrcT0*
Truncate( *

DstT0
i
gradients/Mean_1_grad/truedivRealDivgradients/Mean_1_grad/Tilegradients/Mean_1_grad/Cast*
T0
a
gradients/stack_grad/unstackUnpackgradients/Sum_2_grad/Tile*
T0*	
num*

axis
A
gradients/mul_2_grad/ShapeShapeExp*
T0*
out_type0
J
gradients/mul_2_grad/Shape_1Shape
ExpandDims*
T0*
out_type0
�
*gradients/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_2_grad/Shapegradients/mul_2_grad/Shape_1*
T0
e
gradients/mul_2_grad/MulMul/gradients/Minimum_grad/tuple/control_dependency
ExpandDims*
T0
�
gradients/mul_2_grad/SumSumgradients/mul_2_grad/Mul*gradients/mul_2_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
t
gradients/mul_2_grad/ReshapeReshapegradients/mul_2_grad/Sumgradients/mul_2_grad/Shape*
T0*
Tshape0
`
gradients/mul_2_grad/Mul_1MulExp/gradients/Minimum_grad/tuple/control_dependency*
T0
�
gradients/mul_2_grad/Sum_1Sumgradients/mul_2_grad/Mul_1,gradients/mul_2_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
z
gradients/mul_2_grad/Reshape_1Reshapegradients/mul_2_grad/Sum_1gradients/mul_2_grad/Shape_1*
T0*
Tshape0
m
%gradients/mul_2_grad/tuple/group_depsNoOp^gradients/mul_2_grad/Reshape^gradients/mul_2_grad/Reshape_1
�
-gradients/mul_2_grad/tuple/control_dependencyIdentitygradients/mul_2_grad/Reshape&^gradients/mul_2_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/mul_2_grad/Reshape
�
/gradients/mul_2_grad/tuple/control_dependency_1Identitygradients/mul_2_grad/Reshape_1&^gradients/mul_2_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/mul_2_grad/Reshape_1
M
gradients/mul_3_grad/ShapeShapeclip_by_value_1*
T0*
out_type0
J
gradients/mul_3_grad/Shape_1Shape
ExpandDims*
T0*
out_type0
�
*gradients/mul_3_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_3_grad/Shapegradients/mul_3_grad/Shape_1*
T0
g
gradients/mul_3_grad/MulMul1gradients/Minimum_grad/tuple/control_dependency_1
ExpandDims*
T0
�
gradients/mul_3_grad/SumSumgradients/mul_3_grad/Mul*gradients/mul_3_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
t
gradients/mul_3_grad/ReshapeReshapegradients/mul_3_grad/Sumgradients/mul_3_grad/Shape*
T0*
Tshape0
n
gradients/mul_3_grad/Mul_1Mulclip_by_value_11gradients/Minimum_grad/tuple/control_dependency_1*
T0
�
gradients/mul_3_grad/Sum_1Sumgradients/mul_3_grad/Mul_1,gradients/mul_3_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
z
gradients/mul_3_grad/Reshape_1Reshapegradients/mul_3_grad/Sum_1gradients/mul_3_grad/Shape_1*
T0*
Tshape0
m
%gradients/mul_3_grad/tuple/group_depsNoOp^gradients/mul_3_grad/Reshape^gradients/mul_3_grad/Reshape_1
�
-gradients/mul_3_grad/tuple/control_dependencyIdentitygradients/mul_3_grad/Reshape&^gradients/mul_3_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/mul_3_grad/Reshape
�
/gradients/mul_3_grad/tuple/control_dependency_1Identitygradients/mul_3_grad/Reshape_1&^gradients/mul_3_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/mul_3_grad/Reshape_1
>
gradients/zeros_like_2	ZerosLikeDynamicPartition*
T0
M
%gradients/DynamicPartition_grad/ShapeShapeCast*
T0*
out_type0
S
%gradients/DynamicPartition_grad/ConstConst*
valueB: *
dtype0
�
$gradients/DynamicPartition_grad/ProdProd%gradients/DynamicPartition_grad/Shape%gradients/DynamicPartition_grad/Const*

Tidx0*
	keep_dims( *
T0
U
+gradients/DynamicPartition_grad/range/startConst*
value	B : *
dtype0
U
+gradients/DynamicPartition_grad/range/deltaConst*
value	B :*
dtype0
�
%gradients/DynamicPartition_grad/rangeRange+gradients/DynamicPartition_grad/range/start$gradients/DynamicPartition_grad/Prod+gradients/DynamicPartition_grad/range/delta*

Tidx0
�
'gradients/DynamicPartition_grad/ReshapeReshape%gradients/DynamicPartition_grad/range%gradients/DynamicPartition_grad/Shape*
T0*
Tshape0
�
0gradients/DynamicPartition_grad/DynamicPartitionDynamicPartition'gradients/DynamicPartition_grad/ReshapeCast*
num_partitions*
T0
�
-gradients/DynamicPartition_grad/DynamicStitchDynamicStitch0gradients/DynamicPartition_grad/DynamicPartition2gradients/DynamicPartition_grad/DynamicPartition:1gradients/zeros_like_2gradients/Mean_1_grad/truediv*
T0*
N
R
'gradients/DynamicPartition_grad/Shape_1ShapeMaximum*
T0*
out_type0
�
)gradients/DynamicPartition_grad/Reshape_1Reshape-gradients/DynamicPartition_grad/DynamicStitch'gradients/DynamicPartition_grad/Shape_1*
T0*
Tshape0
�
@gradients/softmax_cross_entropy_with_logits/Reshape_2_grad/ShapeShape!softmax_cross_entropy_with_logits*
T0*
out_type0
�
Bgradients/softmax_cross_entropy_with_logits/Reshape_2_grad/ReshapeReshapegradients/stack_grad/unstack@gradients/softmax_cross_entropy_with_logits/Reshape_2_grad/Shape*
T0*
Tshape0
_
$gradients/clip_by_value_1_grad/ShapeShapeclip_by_value_1/Minimum*
T0*
out_type0
O
&gradients/clip_by_value_1_grad/Shape_1Const*
valueB *
dtype0
w
&gradients/clip_by_value_1_grad/Shape_2Shape-gradients/mul_3_grad/tuple/control_dependency*
T0*
out_type0
W
*gradients/clip_by_value_1_grad/zeros/ConstConst*
valueB
 *    *
dtype0
�
$gradients/clip_by_value_1_grad/zerosFill&gradients/clip_by_value_1_grad/Shape_2*gradients/clip_by_value_1_grad/zeros/Const*
T0*

index_type0
d
+gradients/clip_by_value_1_grad/GreaterEqualGreaterEqualclip_by_value_1/Minimumsub_2*
T0
�
4gradients/clip_by_value_1_grad/BroadcastGradientArgsBroadcastGradientArgs$gradients/clip_by_value_1_grad/Shape&gradients/clip_by_value_1_grad/Shape_1*
T0
�
%gradients/clip_by_value_1_grad/SelectSelect+gradients/clip_by_value_1_grad/GreaterEqual-gradients/mul_3_grad/tuple/control_dependency$gradients/clip_by_value_1_grad/zeros*
T0
�
"gradients/clip_by_value_1_grad/SumSum%gradients/clip_by_value_1_grad/Select4gradients/clip_by_value_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
�
&gradients/clip_by_value_1_grad/ReshapeReshape"gradients/clip_by_value_1_grad/Sum$gradients/clip_by_value_1_grad/Shape*
T0*
Tshape0
�
'gradients/clip_by_value_1_grad/Select_1Select+gradients/clip_by_value_1_grad/GreaterEqual$gradients/clip_by_value_1_grad/zeros-gradients/mul_3_grad/tuple/control_dependency*
T0
�
$gradients/clip_by_value_1_grad/Sum_1Sum'gradients/clip_by_value_1_grad/Select_16gradients/clip_by_value_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
�
(gradients/clip_by_value_1_grad/Reshape_1Reshape$gradients/clip_by_value_1_grad/Sum_1&gradients/clip_by_value_1_grad/Shape_1*
T0*
Tshape0
�
/gradients/clip_by_value_1_grad/tuple/group_depsNoOp'^gradients/clip_by_value_1_grad/Reshape)^gradients/clip_by_value_1_grad/Reshape_1
�
7gradients/clip_by_value_1_grad/tuple/control_dependencyIdentity&gradients/clip_by_value_1_grad/Reshape0^gradients/clip_by_value_1_grad/tuple/group_deps*
T0*9
_class/
-+loc:@gradients/clip_by_value_1_grad/Reshape
�
9gradients/clip_by_value_1_grad/tuple/control_dependency_1Identity(gradients/clip_by_value_1_grad/Reshape_10^gradients/clip_by_value_1_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients/clip_by_value_1_grad/Reshape_1
Q
gradients/Maximum_grad/ShapeShapeSquaredDifference*
T0*
out_type0
U
gradients/Maximum_grad/Shape_1ShapeSquaredDifference_1*
T0*
out_type0
k
gradients/Maximum_grad/Shape_2Shape)gradients/DynamicPartition_grad/Reshape_1*
T0*
out_type0
O
"gradients/Maximum_grad/zeros/ConstConst*
valueB
 *    *
dtype0
�
gradients/Maximum_grad/zerosFillgradients/Maximum_grad/Shape_2"gradients/Maximum_grad/zeros/Const*
T0*

index_type0
d
#gradients/Maximum_grad/GreaterEqualGreaterEqualSquaredDifferenceSquaredDifference_1*
T0
�
,gradients/Maximum_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Maximum_grad/Shapegradients/Maximum_grad/Shape_1*
T0
�
gradients/Maximum_grad/SelectSelect#gradients/Maximum_grad/GreaterEqual)gradients/DynamicPartition_grad/Reshape_1gradients/Maximum_grad/zeros*
T0
�
gradients/Maximum_grad/SumSumgradients/Maximum_grad/Select,gradients/Maximum_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
z
gradients/Maximum_grad/ReshapeReshapegradients/Maximum_grad/Sumgradients/Maximum_grad/Shape*
T0*
Tshape0
�
gradients/Maximum_grad/Select_1Select#gradients/Maximum_grad/GreaterEqualgradients/Maximum_grad/zeros)gradients/DynamicPartition_grad/Reshape_1*
T0
�
gradients/Maximum_grad/Sum_1Sumgradients/Maximum_grad/Select_1.gradients/Maximum_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
�
 gradients/Maximum_grad/Reshape_1Reshapegradients/Maximum_grad/Sum_1gradients/Maximum_grad/Shape_1*
T0*
Tshape0
s
'gradients/Maximum_grad/tuple/group_depsNoOp^gradients/Maximum_grad/Reshape!^gradients/Maximum_grad/Reshape_1
�
/gradients/Maximum_grad/tuple/control_dependencyIdentitygradients/Maximum_grad/Reshape(^gradients/Maximum_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/Maximum_grad/Reshape
�
1gradients/Maximum_grad/tuple/control_dependency_1Identity gradients/Maximum_grad/Reshape_1(^gradients/Maximum_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients/Maximum_grad/Reshape_1
Q
gradients/zeros_like_3	ZerosLike#softmax_cross_entropy_with_logits:1*
T0
r
?gradients/softmax_cross_entropy_with_logits_grad/ExpandDims/dimConst*
valueB :
���������*
dtype0
�
;gradients/softmax_cross_entropy_with_logits_grad/ExpandDims
ExpandDimsBgradients/softmax_cross_entropy_with_logits/Reshape_2_grad/Reshape?gradients/softmax_cross_entropy_with_logits_grad/ExpandDims/dim*

Tdim0*
T0
�
4gradients/softmax_cross_entropy_with_logits_grad/mulMul;gradients/softmax_cross_entropy_with_logits_grad/ExpandDims#softmax_cross_entropy_with_logits:1*
T0
}
;gradients/softmax_cross_entropy_with_logits_grad/LogSoftmax
LogSoftmax)softmax_cross_entropy_with_logits/Reshape*
T0
�
4gradients/softmax_cross_entropy_with_logits_grad/NegNeg;gradients/softmax_cross_entropy_with_logits_grad/LogSoftmax*
T0
t
Agradients/softmax_cross_entropy_with_logits_grad/ExpandDims_1/dimConst*
valueB :
���������*
dtype0
�
=gradients/softmax_cross_entropy_with_logits_grad/ExpandDims_1
ExpandDimsBgradients/softmax_cross_entropy_with_logits/Reshape_2_grad/ReshapeAgradients/softmax_cross_entropy_with_logits_grad/ExpandDims_1/dim*

Tdim0*
T0
�
6gradients/softmax_cross_entropy_with_logits_grad/mul_1Mul=gradients/softmax_cross_entropy_with_logits_grad/ExpandDims_14gradients/softmax_cross_entropy_with_logits_grad/Neg*
T0
�
Agradients/softmax_cross_entropy_with_logits_grad/tuple/group_depsNoOp5^gradients/softmax_cross_entropy_with_logits_grad/mul7^gradients/softmax_cross_entropy_with_logits_grad/mul_1
�
Igradients/softmax_cross_entropy_with_logits_grad/tuple/control_dependencyIdentity4gradients/softmax_cross_entropy_with_logits_grad/mulB^gradients/softmax_cross_entropy_with_logits_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients/softmax_cross_entropy_with_logits_grad/mul
�
Kgradients/softmax_cross_entropy_with_logits_grad/tuple/control_dependency_1Identity6gradients/softmax_cross_entropy_with_logits_grad/mul_1B^gradients/softmax_cross_entropy_with_logits_grad/tuple/group_deps*
T0*I
_class?
=;loc:@gradients/softmax_cross_entropy_with_logits_grad/mul_1
S
,gradients/clip_by_value_1/Minimum_grad/ShapeShapeExp*
T0*
out_type0
W
.gradients/clip_by_value_1/Minimum_grad/Shape_1Const*
valueB *
dtype0
�
.gradients/clip_by_value_1/Minimum_grad/Shape_2Shape7gradients/clip_by_value_1_grad/tuple/control_dependency*
T0*
out_type0
_
2gradients/clip_by_value_1/Minimum_grad/zeros/ConstConst*
valueB
 *    *
dtype0
�
,gradients/clip_by_value_1/Minimum_grad/zerosFill.gradients/clip_by_value_1/Minimum_grad/Shape_22gradients/clip_by_value_1/Minimum_grad/zeros/Const*
T0*

index_type0
R
0gradients/clip_by_value_1/Minimum_grad/LessEqual	LessEqualExpadd_8*
T0
�
<gradients/clip_by_value_1/Minimum_grad/BroadcastGradientArgsBroadcastGradientArgs,gradients/clip_by_value_1/Minimum_grad/Shape.gradients/clip_by_value_1/Minimum_grad/Shape_1*
T0
�
-gradients/clip_by_value_1/Minimum_grad/SelectSelect0gradients/clip_by_value_1/Minimum_grad/LessEqual7gradients/clip_by_value_1_grad/tuple/control_dependency,gradients/clip_by_value_1/Minimum_grad/zeros*
T0
�
*gradients/clip_by_value_1/Minimum_grad/SumSum-gradients/clip_by_value_1/Minimum_grad/Select<gradients/clip_by_value_1/Minimum_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
�
.gradients/clip_by_value_1/Minimum_grad/ReshapeReshape*gradients/clip_by_value_1/Minimum_grad/Sum,gradients/clip_by_value_1/Minimum_grad/Shape*
T0*
Tshape0
�
/gradients/clip_by_value_1/Minimum_grad/Select_1Select0gradients/clip_by_value_1/Minimum_grad/LessEqual,gradients/clip_by_value_1/Minimum_grad/zeros7gradients/clip_by_value_1_grad/tuple/control_dependency*
T0
�
,gradients/clip_by_value_1/Minimum_grad/Sum_1Sum/gradients/clip_by_value_1/Minimum_grad/Select_1>gradients/clip_by_value_1/Minimum_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
�
0gradients/clip_by_value_1/Minimum_grad/Reshape_1Reshape,gradients/clip_by_value_1/Minimum_grad/Sum_1.gradients/clip_by_value_1/Minimum_grad/Shape_1*
T0*
Tshape0
�
7gradients/clip_by_value_1/Minimum_grad/tuple/group_depsNoOp/^gradients/clip_by_value_1/Minimum_grad/Reshape1^gradients/clip_by_value_1/Minimum_grad/Reshape_1
�
?gradients/clip_by_value_1/Minimum_grad/tuple/control_dependencyIdentity.gradients/clip_by_value_1/Minimum_grad/Reshape8^gradients/clip_by_value_1/Minimum_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/clip_by_value_1/Minimum_grad/Reshape
�
Agradients/clip_by_value_1/Minimum_grad/tuple/control_dependency_1Identity0gradients/clip_by_value_1/Minimum_grad/Reshape_18^gradients/clip_by_value_1/Minimum_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/clip_by_value_1/Minimum_grad/Reshape_1
�
'gradients/SquaredDifference_grad/scalarConst0^gradients/Maximum_grad/tuple/control_dependency*
valueB
 *   @*
dtype0
�
$gradients/SquaredDifference_grad/MulMul'gradients/SquaredDifference_grad/scalar/gradients/Maximum_grad/tuple/control_dependency*
T0
�
$gradients/SquaredDifference_grad/subSubextrinsic_returnsSum_60^gradients/Maximum_grad/tuple/control_dependency*
T0
�
&gradients/SquaredDifference_grad/mul_1Mul$gradients/SquaredDifference_grad/Mul$gradients/SquaredDifference_grad/sub*
T0
[
&gradients/SquaredDifference_grad/ShapeShapeextrinsic_returns*
T0*
out_type0
Q
(gradients/SquaredDifference_grad/Shape_1ShapeSum_6*
T0*
out_type0
�
6gradients/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgs&gradients/SquaredDifference_grad/Shape(gradients/SquaredDifference_grad/Shape_1*
T0
�
$gradients/SquaredDifference_grad/SumSum&gradients/SquaredDifference_grad/mul_16gradients/SquaredDifference_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
�
(gradients/SquaredDifference_grad/ReshapeReshape$gradients/SquaredDifference_grad/Sum&gradients/SquaredDifference_grad/Shape*
T0*
Tshape0
�
&gradients/SquaredDifference_grad/Sum_1Sum&gradients/SquaredDifference_grad/mul_18gradients/SquaredDifference_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
�
*gradients/SquaredDifference_grad/Reshape_1Reshape&gradients/SquaredDifference_grad/Sum_1(gradients/SquaredDifference_grad/Shape_1*
T0*
Tshape0
`
$gradients/SquaredDifference_grad/NegNeg*gradients/SquaredDifference_grad/Reshape_1*
T0
�
1gradients/SquaredDifference_grad/tuple/group_depsNoOp%^gradients/SquaredDifference_grad/Neg)^gradients/SquaredDifference_grad/Reshape
�
9gradients/SquaredDifference_grad/tuple/control_dependencyIdentity(gradients/SquaredDifference_grad/Reshape2^gradients/SquaredDifference_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients/SquaredDifference_grad/Reshape
�
;gradients/SquaredDifference_grad/tuple/control_dependency_1Identity$gradients/SquaredDifference_grad/Neg2^gradients/SquaredDifference_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/SquaredDifference_grad/Neg
�
)gradients/SquaredDifference_1_grad/scalarConst2^gradients/Maximum_grad/tuple/control_dependency_1*
valueB
 *   @*
dtype0
�
&gradients/SquaredDifference_1_grad/MulMul)gradients/SquaredDifference_1_grad/scalar1gradients/Maximum_grad/tuple/control_dependency_1*
T0
�
&gradients/SquaredDifference_1_grad/subSubextrinsic_returnsadd_72^gradients/Maximum_grad/tuple/control_dependency_1*
T0
�
(gradients/SquaredDifference_1_grad/mul_1Mul&gradients/SquaredDifference_1_grad/Mul&gradients/SquaredDifference_1_grad/sub*
T0
]
(gradients/SquaredDifference_1_grad/ShapeShapeextrinsic_returns*
T0*
out_type0
S
*gradients/SquaredDifference_1_grad/Shape_1Shapeadd_7*
T0*
out_type0
�
8gradients/SquaredDifference_1_grad/BroadcastGradientArgsBroadcastGradientArgs(gradients/SquaredDifference_1_grad/Shape*gradients/SquaredDifference_1_grad/Shape_1*
T0
�
&gradients/SquaredDifference_1_grad/SumSum(gradients/SquaredDifference_1_grad/mul_18gradients/SquaredDifference_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
�
*gradients/SquaredDifference_1_grad/ReshapeReshape&gradients/SquaredDifference_1_grad/Sum(gradients/SquaredDifference_1_grad/Shape*
T0*
Tshape0
�
(gradients/SquaredDifference_1_grad/Sum_1Sum(gradients/SquaredDifference_1_grad/mul_1:gradients/SquaredDifference_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
�
,gradients/SquaredDifference_1_grad/Reshape_1Reshape(gradients/SquaredDifference_1_grad/Sum_1*gradients/SquaredDifference_1_grad/Shape_1*
T0*
Tshape0
d
&gradients/SquaredDifference_1_grad/NegNeg,gradients/SquaredDifference_1_grad/Reshape_1*
T0
�
3gradients/SquaredDifference_1_grad/tuple/group_depsNoOp'^gradients/SquaredDifference_1_grad/Neg+^gradients/SquaredDifference_1_grad/Reshape
�
;gradients/SquaredDifference_1_grad/tuple/control_dependencyIdentity*gradients/SquaredDifference_1_grad/Reshape4^gradients/SquaredDifference_1_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients/SquaredDifference_1_grad/Reshape
�
=gradients/SquaredDifference_1_grad/tuple/control_dependency_1Identity&gradients/SquaredDifference_1_grad/Neg4^gradients/SquaredDifference_1_grad/tuple/group_deps*
T0*9
_class/
-+loc:@gradients/SquaredDifference_1_grad/Neg
q
>gradients/softmax_cross_entropy_with_logits/Reshape_grad/ShapeShapestrided_slice_8*
T0*
out_type0
�
@gradients/softmax_cross_entropy_with_logits/Reshape_grad/ReshapeReshapeIgradients/softmax_cross_entropy_with_logits_grad/tuple/control_dependency>gradients/softmax_cross_entropy_with_logits/Reshape_grad/Shape*
T0*
Tshape0
m
@gradients/softmax_cross_entropy_with_logits/Reshape_1_grad/ShapeShape	Softmax_2*
T0*
out_type0
�
Bgradients/softmax_cross_entropy_with_logits/Reshape_1_grad/ReshapeReshapeKgradients/softmax_cross_entropy_with_logits_grad/tuple/control_dependency_1@gradients/softmax_cross_entropy_with_logits/Reshape_1_grad/Shape*
T0*
Tshape0
�
gradients/AddNAddN-gradients/mul_2_grad/tuple/control_dependency?gradients/clip_by_value_1/Minimum_grad/tuple/control_dependency*
T0*/
_class%
#!loc:@gradients/mul_2_grad/Reshape*
N
;
gradients/Exp_grad/mulMulgradients/AddNExp*
T0
U
gradients/Sum_6_grad/ShapeShapeextrinsic_value/BiasAdd*
T0*
out_type0
r
gradients/Sum_6_grad/SizeConst*-
_class#
!loc:@gradients/Sum_6_grad/Shape*
value	B :*
dtype0
�
gradients/Sum_6_grad/addAddV2Sum_6/reduction_indicesgradients/Sum_6_grad/Size*
T0*-
_class#
!loc:@gradients/Sum_6_grad/Shape
�
gradients/Sum_6_grad/modFloorModgradients/Sum_6_grad/addgradients/Sum_6_grad/Size*
T0*-
_class#
!loc:@gradients/Sum_6_grad/Shape
t
gradients/Sum_6_grad/Shape_1Const*-
_class#
!loc:@gradients/Sum_6_grad/Shape*
valueB *
dtype0
y
 gradients/Sum_6_grad/range/startConst*-
_class#
!loc:@gradients/Sum_6_grad/Shape*
value	B : *
dtype0
y
 gradients/Sum_6_grad/range/deltaConst*-
_class#
!loc:@gradients/Sum_6_grad/Shape*
value	B :*
dtype0
�
gradients/Sum_6_grad/rangeRange gradients/Sum_6_grad/range/startgradients/Sum_6_grad/Size gradients/Sum_6_grad/range/delta*

Tidx0*-
_class#
!loc:@gradients/Sum_6_grad/Shape
x
gradients/Sum_6_grad/Fill/valueConst*-
_class#
!loc:@gradients/Sum_6_grad/Shape*
value	B :*
dtype0
�
gradients/Sum_6_grad/FillFillgradients/Sum_6_grad/Shape_1gradients/Sum_6_grad/Fill/value*
T0*-
_class#
!loc:@gradients/Sum_6_grad/Shape*

index_type0
�
"gradients/Sum_6_grad/DynamicStitchDynamicStitchgradients/Sum_6_grad/rangegradients/Sum_6_grad/modgradients/Sum_6_grad/Shapegradients/Sum_6_grad/Fill*
T0*-
_class#
!loc:@gradients/Sum_6_grad/Shape*
N
w
gradients/Sum_6_grad/Maximum/yConst*-
_class#
!loc:@gradients/Sum_6_grad/Shape*
value	B :*
dtype0
�
gradients/Sum_6_grad/MaximumMaximum"gradients/Sum_6_grad/DynamicStitchgradients/Sum_6_grad/Maximum/y*
T0*-
_class#
!loc:@gradients/Sum_6_grad/Shape
�
gradients/Sum_6_grad/floordivFloorDivgradients/Sum_6_grad/Shapegradients/Sum_6_grad/Maximum*
T0*-
_class#
!loc:@gradients/Sum_6_grad/Shape
�
gradients/Sum_6_grad/ReshapeReshape;gradients/SquaredDifference_grad/tuple/control_dependency_1"gradients/Sum_6_grad/DynamicStitch*
T0*
Tshape0
y
gradients/Sum_6_grad/TileTilegradients/Sum_6_grad/Reshapegradients/Sum_6_grad/floordiv*

Tmultiples0*
T0
V
gradients/add_7_grad/ShapeShapeextrinsic_value_estimate*
T0*
out_type0
M
gradients/add_7_grad/Shape_1Shapeclip_by_value*
T0*
out_type0
�
*gradients/add_7_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_7_grad/Shapegradients/add_7_grad/Shape_1*
T0
�
gradients/add_7_grad/SumSum=gradients/SquaredDifference_1_grad/tuple/control_dependency_1*gradients/add_7_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
t
gradients/add_7_grad/ReshapeReshapegradients/add_7_grad/Sumgradients/add_7_grad/Shape*
T0*
Tshape0
�
gradients/add_7_grad/Sum_1Sum=gradients/SquaredDifference_1_grad/tuple/control_dependency_1,gradients/add_7_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
z
gradients/add_7_grad/Reshape_1Reshapegradients/add_7_grad/Sum_1gradients/add_7_grad/Shape_1*
T0*
Tshape0
m
%gradients/add_7_grad/tuple/group_depsNoOp^gradients/add_7_grad/Reshape^gradients/add_7_grad/Reshape_1
�
-gradients/add_7_grad/tuple/control_dependencyIdentitygradients/add_7_grad/Reshape&^gradients/add_7_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_7_grad/Reshape
�
/gradients/add_7_grad/tuple/control_dependency_1Identitygradients/add_7_grad/Reshape_1&^gradients/add_7_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/add_7_grad/Reshape_1
a
$gradients/strided_slice_8_grad/ShapeShapeaction_probs/action_probs*
T0*
out_type0
�
/gradients/strided_slice_8_grad/StridedSliceGradStridedSliceGrad$gradients/strided_slice_8_grad/Shapestrided_slice_8/stackstrided_slice_8/stack_1strided_slice_8/stack_2@gradients/softmax_cross_entropy_with_logits/Reshape_grad/Reshape*
Index0*
T0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask
{
gradients/Softmax_2_grad/mulMulBgradients/softmax_cross_entropy_with_logits/Reshape_1_grad/Reshape	Softmax_2*
T0
a
.gradients/Softmax_2_grad/Sum/reduction_indicesConst*
valueB :
���������*
dtype0
�
gradients/Softmax_2_grad/SumSumgradients/Softmax_2_grad/mul.gradients/Softmax_2_grad/Sum/reduction_indices*

Tidx0*
	keep_dims(*
T0
�
gradients/Softmax_2_grad/subSubBgradients/softmax_cross_entropy_with_logits/Reshape_1_grad/Reshapegradients/Softmax_2_grad/Sum*
T0
W
gradients/Softmax_2_grad/mul_1Mulgradients/Softmax_2_grad/sub	Softmax_2*
T0
C
gradients/sub_1_grad/ShapeShapeSum_3*
T0*
out_type0
E
gradients/sub_1_grad/Shape_1ShapeSum_4*
T0*
out_type0
�
*gradients/sub_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/sub_1_grad/Shapegradients/sub_1_grad/Shape_1*
T0
�
gradients/sub_1_grad/SumSumgradients/Exp_grad/mul*gradients/sub_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
t
gradients/sub_1_grad/ReshapeReshapegradients/sub_1_grad/Sumgradients/sub_1_grad/Shape*
T0*
Tshape0
@
gradients/sub_1_grad/NegNeggradients/Exp_grad/mul*
T0
�
gradients/sub_1_grad/Sum_1Sumgradients/sub_1_grad/Neg,gradients/sub_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
z
gradients/sub_1_grad/Reshape_1Reshapegradients/sub_1_grad/Sum_1gradients/sub_1_grad/Shape_1*
T0*
Tshape0
m
%gradients/sub_1_grad/tuple/group_depsNoOp^gradients/sub_1_grad/Reshape^gradients/sub_1_grad/Reshape_1
�
-gradients/sub_1_grad/tuple/control_dependencyIdentitygradients/sub_1_grad/Reshape&^gradients/sub_1_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/sub_1_grad/Reshape
�
/gradients/sub_1_grad/tuple/control_dependency_1Identitygradients/sub_1_grad/Reshape_1&^gradients/sub_1_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/sub_1_grad/Reshape_1
[
"gradients/clip_by_value_grad/ShapeShapeclip_by_value/Minimum*
T0*
out_type0
M
$gradients/clip_by_value_grad/Shape_1Const*
valueB *
dtype0
w
$gradients/clip_by_value_grad/Shape_2Shape/gradients/add_7_grad/tuple/control_dependency_1*
T0*
out_type0
U
(gradients/clip_by_value_grad/zeros/ConstConst*
valueB
 *    *
dtype0
�
"gradients/clip_by_value_grad/zerosFill$gradients/clip_by_value_grad/Shape_2(gradients/clip_by_value_grad/zeros/Const*
T0*

index_type0
`
)gradients/clip_by_value_grad/GreaterEqualGreaterEqualclip_by_value/MinimumNeg_2*
T0
�
2gradients/clip_by_value_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients/clip_by_value_grad/Shape$gradients/clip_by_value_grad/Shape_1*
T0
�
#gradients/clip_by_value_grad/SelectSelect)gradients/clip_by_value_grad/GreaterEqual/gradients/add_7_grad/tuple/control_dependency_1"gradients/clip_by_value_grad/zeros*
T0
�
 gradients/clip_by_value_grad/SumSum#gradients/clip_by_value_grad/Select2gradients/clip_by_value_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
�
$gradients/clip_by_value_grad/ReshapeReshape gradients/clip_by_value_grad/Sum"gradients/clip_by_value_grad/Shape*
T0*
Tshape0
�
%gradients/clip_by_value_grad/Select_1Select)gradients/clip_by_value_grad/GreaterEqual"gradients/clip_by_value_grad/zeros/gradients/add_7_grad/tuple/control_dependency_1*
T0
�
"gradients/clip_by_value_grad/Sum_1Sum%gradients/clip_by_value_grad/Select_14gradients/clip_by_value_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
�
&gradients/clip_by_value_grad/Reshape_1Reshape"gradients/clip_by_value_grad/Sum_1$gradients/clip_by_value_grad/Shape_1*
T0*
Tshape0
�
-gradients/clip_by_value_grad/tuple/group_depsNoOp%^gradients/clip_by_value_grad/Reshape'^gradients/clip_by_value_grad/Reshape_1
�
5gradients/clip_by_value_grad/tuple/control_dependencyIdentity$gradients/clip_by_value_grad/Reshape.^gradients/clip_by_value_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/clip_by_value_grad/Reshape
�
7gradients/clip_by_value_grad/tuple/control_dependency_1Identity&gradients/clip_by_value_grad/Reshape_1.^gradients/clip_by_value_grad/tuple/group_deps*
T0*9
_class/
-+loc:@gradients/clip_by_value_grad/Reshape_1
a
$gradients/strided_slice_7_grad/ShapeShapeaction_probs/action_probs*
T0*
out_type0
�
/gradients/strided_slice_7_grad/StridedSliceGradStridedSliceGrad$gradients/strided_slice_7_grad/Shapestrided_slice_7/stackstrided_slice_7/stack_1strided_slice_7/stack_2gradients/Softmax_2_grad/mul_1*
Index0*
T0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask
E
gradients/Sum_3_grad/ShapeShapestack_1*
T0*
out_type0
r
gradients/Sum_3_grad/SizeConst*-
_class#
!loc:@gradients/Sum_3_grad/Shape*
value	B :*
dtype0
�
gradients/Sum_3_grad/addAddV2Sum_3/reduction_indicesgradients/Sum_3_grad/Size*
T0*-
_class#
!loc:@gradients/Sum_3_grad/Shape
�
gradients/Sum_3_grad/modFloorModgradients/Sum_3_grad/addgradients/Sum_3_grad/Size*
T0*-
_class#
!loc:@gradients/Sum_3_grad/Shape
t
gradients/Sum_3_grad/Shape_1Const*-
_class#
!loc:@gradients/Sum_3_grad/Shape*
valueB *
dtype0
y
 gradients/Sum_3_grad/range/startConst*-
_class#
!loc:@gradients/Sum_3_grad/Shape*
value	B : *
dtype0
y
 gradients/Sum_3_grad/range/deltaConst*-
_class#
!loc:@gradients/Sum_3_grad/Shape*
value	B :*
dtype0
�
gradients/Sum_3_grad/rangeRange gradients/Sum_3_grad/range/startgradients/Sum_3_grad/Size gradients/Sum_3_grad/range/delta*

Tidx0*-
_class#
!loc:@gradients/Sum_3_grad/Shape
x
gradients/Sum_3_grad/Fill/valueConst*-
_class#
!loc:@gradients/Sum_3_grad/Shape*
value	B :*
dtype0
�
gradients/Sum_3_grad/FillFillgradients/Sum_3_grad/Shape_1gradients/Sum_3_grad/Fill/value*
T0*-
_class#
!loc:@gradients/Sum_3_grad/Shape*

index_type0
�
"gradients/Sum_3_grad/DynamicStitchDynamicStitchgradients/Sum_3_grad/rangegradients/Sum_3_grad/modgradients/Sum_3_grad/Shapegradients/Sum_3_grad/Fill*
T0*-
_class#
!loc:@gradients/Sum_3_grad/Shape*
N
w
gradients/Sum_3_grad/Maximum/yConst*-
_class#
!loc:@gradients/Sum_3_grad/Shape*
value	B :*
dtype0
�
gradients/Sum_3_grad/MaximumMaximum"gradients/Sum_3_grad/DynamicStitchgradients/Sum_3_grad/Maximum/y*
T0*-
_class#
!loc:@gradients/Sum_3_grad/Shape
�
gradients/Sum_3_grad/floordivFloorDivgradients/Sum_3_grad/Shapegradients/Sum_3_grad/Maximum*
T0*-
_class#
!loc:@gradients/Sum_3_grad/Shape
�
gradients/Sum_3_grad/ReshapeReshape-gradients/sub_1_grad/tuple/control_dependency"gradients/Sum_3_grad/DynamicStitch*
T0*
Tshape0
y
gradients/Sum_3_grad/TileTilegradients/Sum_3_grad/Reshapegradients/Sum_3_grad/floordiv*

Tmultiples0*
T0
Q
*gradients/clip_by_value/Minimum_grad/ShapeShapesub*
T0*
out_type0
U
,gradients/clip_by_value/Minimum_grad/Shape_1Const*
valueB *
dtype0
�
,gradients/clip_by_value/Minimum_grad/Shape_2Shape5gradients/clip_by_value_grad/tuple/control_dependency*
T0*
out_type0
]
0gradients/clip_by_value/Minimum_grad/zeros/ConstConst*
valueB
 *    *
dtype0
�
*gradients/clip_by_value/Minimum_grad/zerosFill,gradients/clip_by_value/Minimum_grad/Shape_20gradients/clip_by_value/Minimum_grad/zeros/Const*
T0*

index_type0
\
.gradients/clip_by_value/Minimum_grad/LessEqual	LessEqualsubPolynomialDecay_1*
T0
�
:gradients/clip_by_value/Minimum_grad/BroadcastGradientArgsBroadcastGradientArgs*gradients/clip_by_value/Minimum_grad/Shape,gradients/clip_by_value/Minimum_grad/Shape_1*
T0
�
+gradients/clip_by_value/Minimum_grad/SelectSelect.gradients/clip_by_value/Minimum_grad/LessEqual5gradients/clip_by_value_grad/tuple/control_dependency*gradients/clip_by_value/Minimum_grad/zeros*
T0
�
(gradients/clip_by_value/Minimum_grad/SumSum+gradients/clip_by_value/Minimum_grad/Select:gradients/clip_by_value/Minimum_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
�
,gradients/clip_by_value/Minimum_grad/ReshapeReshape(gradients/clip_by_value/Minimum_grad/Sum*gradients/clip_by_value/Minimum_grad/Shape*
T0*
Tshape0
�
-gradients/clip_by_value/Minimum_grad/Select_1Select.gradients/clip_by_value/Minimum_grad/LessEqual*gradients/clip_by_value/Minimum_grad/zeros5gradients/clip_by_value_grad/tuple/control_dependency*
T0
�
*gradients/clip_by_value/Minimum_grad/Sum_1Sum-gradients/clip_by_value/Minimum_grad/Select_1<gradients/clip_by_value/Minimum_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
�
.gradients/clip_by_value/Minimum_grad/Reshape_1Reshape*gradients/clip_by_value/Minimum_grad/Sum_1,gradients/clip_by_value/Minimum_grad/Shape_1*
T0*
Tshape0
�
5gradients/clip_by_value/Minimum_grad/tuple/group_depsNoOp-^gradients/clip_by_value/Minimum_grad/Reshape/^gradients/clip_by_value/Minimum_grad/Reshape_1
�
=gradients/clip_by_value/Minimum_grad/tuple/control_dependencyIdentity,gradients/clip_by_value/Minimum_grad/Reshape6^gradients/clip_by_value/Minimum_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/clip_by_value/Minimum_grad/Reshape
�
?gradients/clip_by_value/Minimum_grad/tuple/control_dependency_1Identity.gradients/clip_by_value/Minimum_grad/Reshape_16^gradients/clip_by_value/Minimum_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/clip_by_value/Minimum_grad/Reshape_1
c
gradients/stack_1_grad/unstackUnpackgradients/Sum_3_grad/Tile*
T0*	
num*

axis
A
gradients/sub_grad/ShapeShapeSum_5*
T0*
out_type0
V
gradients/sub_grad/Shape_1Shapeextrinsic_value_estimate*
T0*
out_type0
�
(gradients/sub_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/sub_grad/Shapegradients/sub_grad/Shape_1*
T0
�
gradients/sub_grad/SumSum=gradients/clip_by_value/Minimum_grad/tuple/control_dependency(gradients/sub_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
n
gradients/sub_grad/ReshapeReshapegradients/sub_grad/Sumgradients/sub_grad/Shape*
T0*
Tshape0
e
gradients/sub_grad/NegNeg=gradients/clip_by_value/Minimum_grad/tuple/control_dependency*
T0
�
gradients/sub_grad/Sum_1Sumgradients/sub_grad/Neg*gradients/sub_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
t
gradients/sub_grad/Reshape_1Reshapegradients/sub_grad/Sum_1gradients/sub_grad/Shape_1*
T0*
Tshape0
g
#gradients/sub_grad/tuple/group_depsNoOp^gradients/sub_grad/Reshape^gradients/sub_grad/Reshape_1
�
+gradients/sub_grad/tuple/control_dependencyIdentitygradients/sub_grad/Reshape$^gradients/sub_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/sub_grad/Reshape
�
-gradients/sub_grad/tuple/control_dependency_1Identitygradients/sub_grad/Reshape_1$^gradients/sub_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/sub_grad/Reshape_1
F
gradients/Neg_grad/NegNeggradients/stack_1_grad/unstack*
T0
U
gradients/Sum_5_grad/ShapeShapeextrinsic_value/BiasAdd*
T0*
out_type0
r
gradients/Sum_5_grad/SizeConst*-
_class#
!loc:@gradients/Sum_5_grad/Shape*
value	B :*
dtype0
�
gradients/Sum_5_grad/addAddV2Sum_5/reduction_indicesgradients/Sum_5_grad/Size*
T0*-
_class#
!loc:@gradients/Sum_5_grad/Shape
�
gradients/Sum_5_grad/modFloorModgradients/Sum_5_grad/addgradients/Sum_5_grad/Size*
T0*-
_class#
!loc:@gradients/Sum_5_grad/Shape
t
gradients/Sum_5_grad/Shape_1Const*-
_class#
!loc:@gradients/Sum_5_grad/Shape*
valueB *
dtype0
y
 gradients/Sum_5_grad/range/startConst*-
_class#
!loc:@gradients/Sum_5_grad/Shape*
value	B : *
dtype0
y
 gradients/Sum_5_grad/range/deltaConst*-
_class#
!loc:@gradients/Sum_5_grad/Shape*
value	B :*
dtype0
�
gradients/Sum_5_grad/rangeRange gradients/Sum_5_grad/range/startgradients/Sum_5_grad/Size gradients/Sum_5_grad/range/delta*

Tidx0*-
_class#
!loc:@gradients/Sum_5_grad/Shape
x
gradients/Sum_5_grad/Fill/valueConst*-
_class#
!loc:@gradients/Sum_5_grad/Shape*
value	B :*
dtype0
�
gradients/Sum_5_grad/FillFillgradients/Sum_5_grad/Shape_1gradients/Sum_5_grad/Fill/value*
T0*-
_class#
!loc:@gradients/Sum_5_grad/Shape*

index_type0
�
"gradients/Sum_5_grad/DynamicStitchDynamicStitchgradients/Sum_5_grad/rangegradients/Sum_5_grad/modgradients/Sum_5_grad/Shapegradients/Sum_5_grad/Fill*
T0*-
_class#
!loc:@gradients/Sum_5_grad/Shape*
N
w
gradients/Sum_5_grad/Maximum/yConst*-
_class#
!loc:@gradients/Sum_5_grad/Shape*
value	B :*
dtype0
�
gradients/Sum_5_grad/MaximumMaximum"gradients/Sum_5_grad/DynamicStitchgradients/Sum_5_grad/Maximum/y*
T0*-
_class#
!loc:@gradients/Sum_5_grad/Shape
�
gradients/Sum_5_grad/floordivFloorDivgradients/Sum_5_grad/Shapegradients/Sum_5_grad/Maximum*
T0*-
_class#
!loc:@gradients/Sum_5_grad/Shape
�
gradients/Sum_5_grad/ReshapeReshape+gradients/sub_grad/tuple/control_dependency"gradients/Sum_5_grad/DynamicStitch*
T0*
Tshape0
y
gradients/Sum_5_grad/TileTilegradients/Sum_5_grad/Reshapegradients/Sum_5_grad/floordiv*

Tmultiples0*
T0
�
Bgradients/softmax_cross_entropy_with_logits_1/Reshape_2_grad/ShapeShape#softmax_cross_entropy_with_logits_1*
T0*
out_type0
�
Dgradients/softmax_cross_entropy_with_logits_1/Reshape_2_grad/ReshapeReshapegradients/Neg_grad/NegBgradients/softmax_cross_entropy_with_logits_1/Reshape_2_grad/Shape*
T0*
Tshape0
�
gradients/AddN_1AddNgradients/Sum_6_grad/Tilegradients/Sum_5_grad/Tile*
T0*,
_class"
 loc:@gradients/Sum_6_grad/Tile*
N
s
2gradients/extrinsic_value/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_1*
T0*
data_formatNHWC
�
7gradients/extrinsic_value/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_13^gradients/extrinsic_value/BiasAdd_grad/BiasAddGrad
�
?gradients/extrinsic_value/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_18^gradients/extrinsic_value/BiasAdd_grad/tuple/group_deps*
T0*,
_class"
 loc:@gradients/Sum_6_grad/Tile
�
Agradients/extrinsic_value/BiasAdd_grad/tuple/control_dependency_1Identity2gradients/extrinsic_value/BiasAdd_grad/BiasAddGrad8^gradients/extrinsic_value/BiasAdd_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/extrinsic_value/BiasAdd_grad/BiasAddGrad
S
gradients/zeros_like_4	ZerosLike%softmax_cross_entropy_with_logits_1:1*
T0
t
Agradients/softmax_cross_entropy_with_logits_1_grad/ExpandDims/dimConst*
valueB :
���������*
dtype0
�
=gradients/softmax_cross_entropy_with_logits_1_grad/ExpandDims
ExpandDimsDgradients/softmax_cross_entropy_with_logits_1/Reshape_2_grad/ReshapeAgradients/softmax_cross_entropy_with_logits_1_grad/ExpandDims/dim*

Tdim0*
T0
�
6gradients/softmax_cross_entropy_with_logits_1_grad/mulMul=gradients/softmax_cross_entropy_with_logits_1_grad/ExpandDims%softmax_cross_entropy_with_logits_1:1*
T0
�
=gradients/softmax_cross_entropy_with_logits_1_grad/LogSoftmax
LogSoftmax+softmax_cross_entropy_with_logits_1/Reshape*
T0
�
6gradients/softmax_cross_entropy_with_logits_1_grad/NegNeg=gradients/softmax_cross_entropy_with_logits_1_grad/LogSoftmax*
T0
v
Cgradients/softmax_cross_entropy_with_logits_1_grad/ExpandDims_1/dimConst*
valueB :
���������*
dtype0
�
?gradients/softmax_cross_entropy_with_logits_1_grad/ExpandDims_1
ExpandDimsDgradients/softmax_cross_entropy_with_logits_1/Reshape_2_grad/ReshapeCgradients/softmax_cross_entropy_with_logits_1_grad/ExpandDims_1/dim*

Tdim0*
T0
�
8gradients/softmax_cross_entropy_with_logits_1_grad/mul_1Mul?gradients/softmax_cross_entropy_with_logits_1_grad/ExpandDims_16gradients/softmax_cross_entropy_with_logits_1_grad/Neg*
T0
�
Cgradients/softmax_cross_entropy_with_logits_1_grad/tuple/group_depsNoOp7^gradients/softmax_cross_entropy_with_logits_1_grad/mul9^gradients/softmax_cross_entropy_with_logits_1_grad/mul_1
�
Kgradients/softmax_cross_entropy_with_logits_1_grad/tuple/control_dependencyIdentity6gradients/softmax_cross_entropy_with_logits_1_grad/mulD^gradients/softmax_cross_entropy_with_logits_1_grad/tuple/group_deps*
T0*I
_class?
=;loc:@gradients/softmax_cross_entropy_with_logits_1_grad/mul
�
Mgradients/softmax_cross_entropy_with_logits_1_grad/tuple/control_dependency_1Identity8gradients/softmax_cross_entropy_with_logits_1_grad/mul_1D^gradients/softmax_cross_entropy_with_logits_1_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients/softmax_cross_entropy_with_logits_1_grad/mul_1
�
,gradients/extrinsic_value/MatMul_grad/MatMulMatMul?gradients/extrinsic_value/BiasAdd_grad/tuple/control_dependencyextrinsic_value/kernel/read*
transpose_b(*
T0*
transpose_a( 
�
.gradients/extrinsic_value/MatMul_grad/MatMul_1MatMul	Reshape_2?gradients/extrinsic_value/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(
�
6gradients/extrinsic_value/MatMul_grad/tuple/group_depsNoOp-^gradients/extrinsic_value/MatMul_grad/MatMul/^gradients/extrinsic_value/MatMul_grad/MatMul_1
�
>gradients/extrinsic_value/MatMul_grad/tuple/control_dependencyIdentity,gradients/extrinsic_value/MatMul_grad/MatMul7^gradients/extrinsic_value/MatMul_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/extrinsic_value/MatMul_grad/MatMul
�
@gradients/extrinsic_value/MatMul_grad/tuple/control_dependency_1Identity.gradients/extrinsic_value/MatMul_grad/MatMul_17^gradients/extrinsic_value/MatMul_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/extrinsic_value/MatMul_grad/MatMul_1
t
@gradients/softmax_cross_entropy_with_logits_1/Reshape_grad/ShapeShapestrided_slice_10*
T0*
out_type0
�
Bgradients/softmax_cross_entropy_with_logits_1/Reshape_grad/ReshapeReshapeKgradients/softmax_cross_entropy_with_logits_1_grad/tuple/control_dependency@gradients/softmax_cross_entropy_with_logits_1/Reshape_grad/Shape*
T0*
Tshape0
X
%gradients/strided_slice_10_grad/ShapeShapeconcat_6/concat*
T0*
out_type0
�
0gradients/strided_slice_10_grad/StridedSliceGradStridedSliceGrad%gradients/strided_slice_10_grad/Shapestrided_slice_10/stackstrided_slice_10/stack_1strided_slice_10/stack_2Bgradients/softmax_cross_entropy_with_logits_1/Reshape_grad/Reshape*
Index0*
T0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask
p
gradients/Log_1_grad/Reciprocal
Reciprocaladd_31^gradients/strided_slice_10_grad/StridedSliceGrad*
T0
{
gradients/Log_1_grad/mulMul0gradients/strided_slice_10_grad/StridedSliceGradgradients/Log_1_grad/Reciprocal*
T0
E
gradients/add_3_grad/ShapeShapetruediv*
T0*
out_type0
G
gradients/add_3_grad/Shape_1Shapeadd_3/y*
T0*
out_type0
�
*gradients/add_3_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_3_grad/Shapegradients/add_3_grad/Shape_1*
T0
�
gradients/add_3_grad/SumSumgradients/Log_1_grad/mul*gradients/add_3_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
t
gradients/add_3_grad/ReshapeReshapegradients/add_3_grad/Sumgradients/add_3_grad/Shape*
T0*
Tshape0
�
gradients/add_3_grad/Sum_1Sumgradients/Log_1_grad/mul,gradients/add_3_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
z
gradients/add_3_grad/Reshape_1Reshapegradients/add_3_grad/Sum_1gradients/add_3_grad/Shape_1*
T0*
Tshape0
m
%gradients/add_3_grad/tuple/group_depsNoOp^gradients/add_3_grad/Reshape^gradients/add_3_grad/Reshape_1
�
-gradients/add_3_grad/tuple/control_dependencyIdentitygradients/add_3_grad/Reshape&^gradients/add_3_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_3_grad/Reshape
�
/gradients/add_3_grad/tuple/control_dependency_1Identitygradients/add_3_grad/Reshape_1&^gradients/add_3_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/add_3_grad/Reshape_1
C
gradients/truediv_grad/ShapeShapeMul*
T0*
out_type0
E
gradients/truediv_grad/Shape_1ShapeSum*
T0*
out_type0
�
,gradients/truediv_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/truediv_grad/Shapegradients/truediv_grad/Shape_1*
T0
f
gradients/truediv_grad/RealDivRealDiv-gradients/add_3_grad/tuple/control_dependencySum*
T0
�
gradients/truediv_grad/SumSumgradients/truediv_grad/RealDiv,gradients/truediv_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
z
gradients/truediv_grad/ReshapeReshapegradients/truediv_grad/Sumgradients/truediv_grad/Shape*
T0*
Tshape0
/
gradients/truediv_grad/NegNegMul*
T0
U
 gradients/truediv_grad/RealDiv_1RealDivgradients/truediv_grad/NegSum*
T0
[
 gradients/truediv_grad/RealDiv_2RealDiv gradients/truediv_grad/RealDiv_1Sum*
T0
{
gradients/truediv_grad/mulMul-gradients/add_3_grad/tuple/control_dependency gradients/truediv_grad/RealDiv_2*
T0
�
gradients/truediv_grad/Sum_1Sumgradients/truediv_grad/mul.gradients/truediv_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
�
 gradients/truediv_grad/Reshape_1Reshapegradients/truediv_grad/Sum_1gradients/truediv_grad/Shape_1*
T0*
Tshape0
s
'gradients/truediv_grad/tuple/group_depsNoOp^gradients/truediv_grad/Reshape!^gradients/truediv_grad/Reshape_1
�
/gradients/truediv_grad/tuple/control_dependencyIdentitygradients/truediv_grad/Reshape(^gradients/truediv_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/truediv_grad/Reshape
�
1gradients/truediv_grad/tuple/control_dependency_1Identity gradients/truediv_grad/Reshape_1(^gradients/truediv_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients/truediv_grad/Reshape_1
?
gradients/Sum_grad/ShapeShapeMul*
T0*
out_type0
n
gradients/Sum_grad/SizeConst*+
_class!
loc:@gradients/Sum_grad/Shape*
value	B :*
dtype0
�
gradients/Sum_grad/addAddV2Sum/reduction_indicesgradients/Sum_grad/Size*
T0*+
_class!
loc:@gradients/Sum_grad/Shape
�
gradients/Sum_grad/modFloorModgradients/Sum_grad/addgradients/Sum_grad/Size*
T0*+
_class!
loc:@gradients/Sum_grad/Shape
p
gradients/Sum_grad/Shape_1Const*+
_class!
loc:@gradients/Sum_grad/Shape*
valueB *
dtype0
u
gradients/Sum_grad/range/startConst*+
_class!
loc:@gradients/Sum_grad/Shape*
value	B : *
dtype0
u
gradients/Sum_grad/range/deltaConst*+
_class!
loc:@gradients/Sum_grad/Shape*
value	B :*
dtype0
�
gradients/Sum_grad/rangeRangegradients/Sum_grad/range/startgradients/Sum_grad/Sizegradients/Sum_grad/range/delta*

Tidx0*+
_class!
loc:@gradients/Sum_grad/Shape
t
gradients/Sum_grad/Fill/valueConst*+
_class!
loc:@gradients/Sum_grad/Shape*
value	B :*
dtype0
�
gradients/Sum_grad/FillFillgradients/Sum_grad/Shape_1gradients/Sum_grad/Fill/value*
T0*+
_class!
loc:@gradients/Sum_grad/Shape*

index_type0
�
 gradients/Sum_grad/DynamicStitchDynamicStitchgradients/Sum_grad/rangegradients/Sum_grad/modgradients/Sum_grad/Shapegradients/Sum_grad/Fill*
T0*+
_class!
loc:@gradients/Sum_grad/Shape*
N
s
gradients/Sum_grad/Maximum/yConst*+
_class!
loc:@gradients/Sum_grad/Shape*
value	B :*
dtype0
�
gradients/Sum_grad/MaximumMaximum gradients/Sum_grad/DynamicStitchgradients/Sum_grad/Maximum/y*
T0*+
_class!
loc:@gradients/Sum_grad/Shape
�
gradients/Sum_grad/floordivFloorDivgradients/Sum_grad/Shapegradients/Sum_grad/Maximum*
T0*+
_class!
loc:@gradients/Sum_grad/Shape
�
gradients/Sum_grad/ReshapeReshape1gradients/truediv_grad/tuple/control_dependency_1 gradients/Sum_grad/DynamicStitch*
T0*
Tshape0
s
gradients/Sum_grad/TileTilegradients/Sum_grad/Reshapegradients/Sum_grad/floordiv*

Tmultiples0*
T0
�
gradients/AddN_2AddN/gradients/truediv_grad/tuple/control_dependencygradients/Sum_grad/Tile*
T0*1
_class'
%#loc:@gradients/truediv_grad/Reshape*
N
A
gradients/Mul_grad/ShapeShapeadd_1*
T0*
out_type0
M
gradients/Mul_grad/Shape_1Shapestrided_slice_3*
T0*
out_type0
�
(gradients/Mul_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Mul_grad/Shapegradients/Mul_grad/Shape_1*
T0
I
gradients/Mul_grad/MulMulgradients/AddN_2strided_slice_3*
T0
�
gradients/Mul_grad/SumSumgradients/Mul_grad/Mul(gradients/Mul_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
n
gradients/Mul_grad/ReshapeReshapegradients/Mul_grad/Sumgradients/Mul_grad/Shape*
T0*
Tshape0
A
gradients/Mul_grad/Mul_1Muladd_1gradients/AddN_2*
T0
�
gradients/Mul_grad/Sum_1Sumgradients/Mul_grad/Mul_1*gradients/Mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
t
gradients/Mul_grad/Reshape_1Reshapegradients/Mul_grad/Sum_1gradients/Mul_grad/Shape_1*
T0*
Tshape0
g
#gradients/Mul_grad/tuple/group_depsNoOp^gradients/Mul_grad/Reshape^gradients/Mul_grad/Reshape_1
�
+gradients/Mul_grad/tuple/control_dependencyIdentitygradients/Mul_grad/Reshape$^gradients/Mul_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients/Mul_grad/Reshape
�
-gradients/Mul_grad/tuple/control_dependency_1Identitygradients/Mul_grad/Reshape_1$^gradients/Mul_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/Mul_grad/Reshape_1
E
gradients/add_1_grad/ShapeShapeSoftmax*
T0*
out_type0
G
gradients/add_1_grad/Shape_1Shapeadd_1/y*
T0*
out_type0
�
*gradients/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_1_grad/Shapegradients/add_1_grad/Shape_1*
T0
�
gradients/add_1_grad/SumSum+gradients/Mul_grad/tuple/control_dependency*gradients/add_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
t
gradients/add_1_grad/ReshapeReshapegradients/add_1_grad/Sumgradients/add_1_grad/Shape*
T0*
Tshape0
�
gradients/add_1_grad/Sum_1Sum+gradients/Mul_grad/tuple/control_dependency,gradients/add_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
z
gradients/add_1_grad/Reshape_1Reshapegradients/add_1_grad/Sum_1gradients/add_1_grad/Shape_1*
T0*
Tshape0
m
%gradients/add_1_grad/tuple/group_depsNoOp^gradients/add_1_grad/Reshape^gradients/add_1_grad/Reshape_1
�
-gradients/add_1_grad/tuple/control_dependencyIdentitygradients/add_1_grad/Reshape&^gradients/add_1_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_1_grad/Reshape
�
/gradients/add_1_grad/tuple/control_dependency_1Identitygradients/add_1_grad/Reshape_1&^gradients/add_1_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients/add_1_grad/Reshape_1
b
gradients/Softmax_grad/mulMul-gradients/add_1_grad/tuple/control_dependencySoftmax*
T0
_
,gradients/Softmax_grad/Sum/reduction_indicesConst*
valueB :
���������*
dtype0
�
gradients/Softmax_grad/SumSumgradients/Softmax_grad/mul,gradients/Softmax_grad/Sum/reduction_indices*

Tidx0*
	keep_dims(*
T0
u
gradients/Softmax_grad/subSub-gradients/add_1_grad/tuple/control_dependencygradients/Softmax_grad/Sum*
T0
Q
gradients/Softmax_grad/mul_1Mulgradients/Softmax_grad/subSoftmax*
T0
a
$gradients/strided_slice_2_grad/ShapeShapeaction_probs/action_probs*
T0*
out_type0
�
/gradients/strided_slice_2_grad/StridedSliceGradStridedSliceGrad$gradients/strided_slice_2_grad/Shapestrided_slice_2/stackstrided_slice_2/stack_1strided_slice_2/stack_2gradients/Softmax_grad/mul_1*
Index0*
T0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask
�
gradients/AddN_3AddN/gradients/strided_slice_8_grad/StridedSliceGrad/gradients/strided_slice_7_grad/StridedSliceGrad/gradients/strided_slice_2_grad/StridedSliceGrad*
T0*B
_class8
64loc:@gradients/strided_slice_8_grad/StridedSliceGrad*
N
�
"gradients/dense/MatMul_grad/MatMulMatMulgradients/AddN_3dense/kernel/read*
transpose_b(*
T0*
transpose_a( 
z
$gradients/dense/MatMul_grad/MatMul_1MatMul	Reshape_2gradients/AddN_3*
transpose_b( *
T0*
transpose_a(
�
,gradients/dense/MatMul_grad/tuple/group_depsNoOp#^gradients/dense/MatMul_grad/MatMul%^gradients/dense/MatMul_grad/MatMul_1
�
4gradients/dense/MatMul_grad/tuple/control_dependencyIdentity"gradients/dense/MatMul_grad/MatMul-^gradients/dense/MatMul_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients/dense/MatMul_grad/MatMul
�
6gradients/dense/MatMul_grad/tuple/control_dependency_1Identity$gradients/dense/MatMul_grad/MatMul_1-^gradients/dense/MatMul_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/dense/MatMul_grad/MatMul_1
�
gradients/AddN_4AddN>gradients/extrinsic_value/MatMul_grad/tuple/control_dependency4gradients/dense/MatMul_grad/tuple/control_dependency*
T0*?
_class5
31loc:@gradients/extrinsic_value/MatMul_grad/MatMul*
N
V
gradients/Reshape_2_grad/ShapeShapelstm/rnn/transpose_1*
T0*
out_type0
t
 gradients/Reshape_2_grad/ReshapeReshapegradients/AddN_4gradients/Reshape_2_grad/Shape*
T0*
Tshape0
f
5gradients/lstm/rnn/transpose_1_grad/InvertPermutationInvertPermutationlstm/rnn/concat_2*
T0
�
-gradients/lstm/rnn/transpose_1_grad/transpose	Transpose gradients/Reshape_2_grad/Reshape5gradients/lstm/rnn/transpose_1_grad/InvertPermutation*
Tperm0*
T0
�
^gradients/lstm/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3lstm/rnn/TensorArraylstm/rnn/while/Exit_2*'
_class
loc:@lstm/rnn/TensorArray*
source	gradients
�
Zgradients/lstm/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/gradient_flowIdentitylstm/rnn/while/Exit_2_^gradients/lstm/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*'
_class
loc:@lstm/rnn/TensorArray
�
dgradients/lstm/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3^gradients/lstm/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3lstm/rnn/TensorArrayStack/range-gradients/lstm/rnn/transpose_1_grad/transposeZgradients/lstm/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/gradient_flow*
T0
C
gradients/zeros_like_5	ZerosLikelstm/rnn/while/Exit_3*
T0
C
gradients/zeros_like_6	ZerosLikelstm/rnn/while/Exit_4*
T0
�
+gradients/lstm/rnn/while/Exit_2_grad/b_exitEnterdgradients/lstm/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayScatter/TensorArrayScatterV3*
T0*
is_constant( *
parallel_iterations *6

frame_name(&gradients/lstm/rnn/while/while_context
�
+gradients/lstm/rnn/while/Exit_3_grad/b_exitEntergradients/zeros_like_5*
T0*
is_constant( *
parallel_iterations *6

frame_name(&gradients/lstm/rnn/while/while_context
�
+gradients/lstm/rnn/while/Exit_4_grad/b_exitEntergradients/zeros_like_6*
T0*
is_constant( *
parallel_iterations *6

frame_name(&gradients/lstm/rnn/while/while_context
�
/gradients/lstm/rnn/while/Switch_2_grad/b_switchMerge+gradients/lstm/rnn/while/Exit_2_grad/b_exit6gradients/lstm/rnn/while/Switch_2_grad_1/NextIteration*
T0*
N
�
/gradients/lstm/rnn/while/Switch_3_grad/b_switchMerge+gradients/lstm/rnn/while/Exit_3_grad/b_exit6gradients/lstm/rnn/while/Switch_3_grad_1/NextIteration*
T0*
N
�
/gradients/lstm/rnn/while/Switch_4_grad/b_switchMerge+gradients/lstm/rnn/while/Exit_4_grad/b_exit6gradients/lstm/rnn/while/Switch_4_grad_1/NextIteration*
T0*
N
�
,gradients/lstm/rnn/while/Merge_2_grad/SwitchSwitch/gradients/lstm/rnn/while/Switch_2_grad/b_switchgradients/b_count_2*
T0*B
_class8
64loc:@gradients/lstm/rnn/while/Switch_2_grad/b_switch
m
6gradients/lstm/rnn/while/Merge_2_grad/tuple/group_depsNoOp-^gradients/lstm/rnn/while/Merge_2_grad/Switch
�
>gradients/lstm/rnn/while/Merge_2_grad/tuple/control_dependencyIdentity,gradients/lstm/rnn/while/Merge_2_grad/Switch7^gradients/lstm/rnn/while/Merge_2_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/lstm/rnn/while/Switch_2_grad/b_switch
�
@gradients/lstm/rnn/while/Merge_2_grad/tuple/control_dependency_1Identity.gradients/lstm/rnn/while/Merge_2_grad/Switch:17^gradients/lstm/rnn/while/Merge_2_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/lstm/rnn/while/Switch_2_grad/b_switch
�
,gradients/lstm/rnn/while/Merge_3_grad/SwitchSwitch/gradients/lstm/rnn/while/Switch_3_grad/b_switchgradients/b_count_2*
T0*B
_class8
64loc:@gradients/lstm/rnn/while/Switch_3_grad/b_switch
m
6gradients/lstm/rnn/while/Merge_3_grad/tuple/group_depsNoOp-^gradients/lstm/rnn/while/Merge_3_grad/Switch
�
>gradients/lstm/rnn/while/Merge_3_grad/tuple/control_dependencyIdentity,gradients/lstm/rnn/while/Merge_3_grad/Switch7^gradients/lstm/rnn/while/Merge_3_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/lstm/rnn/while/Switch_3_grad/b_switch
�
@gradients/lstm/rnn/while/Merge_3_grad/tuple/control_dependency_1Identity.gradients/lstm/rnn/while/Merge_3_grad/Switch:17^gradients/lstm/rnn/while/Merge_3_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/lstm/rnn/while/Switch_3_grad/b_switch
�
,gradients/lstm/rnn/while/Merge_4_grad/SwitchSwitch/gradients/lstm/rnn/while/Switch_4_grad/b_switchgradients/b_count_2*
T0*B
_class8
64loc:@gradients/lstm/rnn/while/Switch_4_grad/b_switch
m
6gradients/lstm/rnn/while/Merge_4_grad/tuple/group_depsNoOp-^gradients/lstm/rnn/while/Merge_4_grad/Switch
�
>gradients/lstm/rnn/while/Merge_4_grad/tuple/control_dependencyIdentity,gradients/lstm/rnn/while/Merge_4_grad/Switch7^gradients/lstm/rnn/while/Merge_4_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/lstm/rnn/while/Switch_4_grad/b_switch
�
@gradients/lstm/rnn/while/Merge_4_grad/tuple/control_dependency_1Identity.gradients/lstm/rnn/while/Merge_4_grad/Switch:17^gradients/lstm/rnn/while/Merge_4_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/lstm/rnn/while/Switch_4_grad/b_switch
{
*gradients/lstm/rnn/while/Enter_2_grad/ExitExit>gradients/lstm/rnn/while/Merge_2_grad/tuple/control_dependency*
T0
{
*gradients/lstm/rnn/while/Enter_3_grad/ExitExit>gradients/lstm/rnn/while/Merge_3_grad/tuple/control_dependency*
T0
{
*gradients/lstm/rnn/while/Enter_4_grad/ExitExit>gradients/lstm/rnn/while/Merge_4_grad/tuple/control_dependency*
T0
�
cgradients/lstm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3igradients/lstm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter@gradients/lstm/rnn/while/Merge_2_grad/tuple/control_dependency_1*7
_class-
+)loc:@lstm/rnn/while/basic_lstm_cell/Mul_2*
source	gradients
�
igradients/lstm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterEnterlstm/rnn/TensorArray*
T0*7
_class-
+)loc:@lstm/rnn/while/basic_lstm_cell/Mul_2*
is_constant(*
parallel_iterations *6

frame_name(&gradients/lstm/rnn/while/while_context
�
_gradients/lstm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/gradient_flowIdentity@gradients/lstm/rnn/while/Merge_2_grad/tuple/control_dependency_1d^gradients/lstm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*7
_class-
+)loc:@lstm/rnn/while/basic_lstm_cell/Mul_2
�
Sgradients/lstm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3TensorArrayReadV3cgradients/lstm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3^gradients/lstm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2_gradients/lstm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/gradient_flow*
dtype0
�
Ygradients/lstm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/ConstConst*,
_class"
 loc:@lstm/rnn/while/Identity_1*
valueB :
���������*
dtype0
�
Ygradients/lstm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_accStackV2Ygradients/lstm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Const*
	elem_type0*,
_class"
 loc:@lstm/rnn/while/Identity_1*

stack_name 
�
Ygradients/lstm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/EnterEnterYgradients/lstm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc*
T0*
is_constant(*
parallel_iterations *,

frame_namelstm/rnn/while/while_context
�
_gradients/lstm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2StackPushV2Ygradients/lstm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Enterlstm/rnn/while/Identity_1^gradients/Add*
T0*
swap_memory( 
�
^gradients/lstm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2
StackPopV2dgradients/lstm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2/Enter^gradients/Sub*
	elem_type0
�
dgradients/lstm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2/EnterEnterYgradients/lstm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc*
T0*
is_constant(*
parallel_iterations *6

frame_name(&gradients/lstm/rnn/while/while_context
�
Zgradients/lstm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/b_syncControlTrigger_^gradients/lstm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2U^gradients/lstm/rnn/while/basic_lstm_cell/Add_1_grad/BroadcastGradientArgs/StackPopV2W^gradients/lstm/rnn/while/basic_lstm_cell/Add_1_grad/BroadcastGradientArgs/StackPopV2_1S^gradients/lstm/rnn/while/basic_lstm_cell/Add_grad/BroadcastGradientArgs/StackPopV2U^gradients/lstm/rnn/while/basic_lstm_cell/Add_grad/BroadcastGradientArgs/StackPopV2_1I^gradients/lstm/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2U^gradients/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/BroadcastGradientArgs/StackPopV2W^gradients/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/BroadcastGradientArgs/StackPopV2_1C^gradients/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/StackPopV2E^gradients/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/StackPopV2U^gradients/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/BroadcastGradientArgs/StackPopV2W^gradients/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/BroadcastGradientArgs/StackPopV2_1C^gradients/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/StackPopV2E^gradients/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/StackPopV2S^gradients/lstm/rnn/while/basic_lstm_cell/Mul_grad/BroadcastGradientArgs/StackPopV2U^gradients/lstm/rnn/while/basic_lstm_cell/Mul_grad/BroadcastGradientArgs/StackPopV2_1A^gradients/lstm/rnn/while/basic_lstm_cell/Mul_grad/Mul/StackPopV2C^gradients/lstm/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/StackPopV2G^gradients/lstm/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPopV2I^gradients/lstm/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPopV2_1
�
Rgradients/lstm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/group_depsNoOpA^gradients/lstm/rnn/while/Merge_2_grad/tuple/control_dependency_1T^gradients/lstm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3
�
Zgradients/lstm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependencyIdentitySgradients/lstm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3S^gradients/lstm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/group_deps*
T0*f
_class\
ZXloc:@gradients/lstm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3
�
\gradients/lstm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency_1Identity@gradients/lstm/rnn/while/Merge_2_grad/tuple/control_dependency_1S^gradients/lstm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/lstm/rnn/while/Switch_2_grad/b_switch
�
gradients/AddN_5AddN@gradients/lstm/rnn/while/Merge_4_grad/tuple/control_dependency_1Zgradients/lstm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency*
T0*B
_class8
64loc:@gradients/lstm/rnn/while/Switch_4_grad/b_switch*
N
�
9gradients/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/ShapeShape%lstm/rnn/while/basic_lstm_cell/Tanh_1*
T0*
out_type0
�
;gradients/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/Shape_1Shape(lstm/rnn/while/basic_lstm_cell/Sigmoid_2*
T0*
out_type0
�
Igradients/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsTgradients/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/BroadcastGradientArgs/StackPopV2Vgradients/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/BroadcastGradientArgs/StackPopV2_1*
T0
�
Ogradients/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/BroadcastGradientArgs/ConstConst*L
_classB
@>loc:@gradients/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/Shape*
valueB :
���������*
dtype0
�
Ogradients/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/BroadcastGradientArgs/f_accStackV2Ogradients/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/BroadcastGradientArgs/Const*
	elem_type0*L
_classB
@>loc:@gradients/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/Shape*

stack_name 
�
Ogradients/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/BroadcastGradientArgs/EnterEnterOgradients/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *,

frame_namelstm/rnn/while/while_context
�
Ugradients/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/BroadcastGradientArgs/StackPushV2StackPushV2Ogradients/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/BroadcastGradientArgs/Enter9gradients/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/Shape^gradients/Add*
T0*
swap_memory( 
�
Tgradients/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/BroadcastGradientArgs/StackPopV2
StackPopV2Zgradients/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub*
	elem_type0
�
Zgradients/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/BroadcastGradientArgs/StackPopV2/EnterEnterOgradients/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *6

frame_name(&gradients/lstm/rnn/while/while_context
�
Qgradients/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/BroadcastGradientArgs/Const_1Const*N
_classD
B@loc:@gradients/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/Shape_1*
valueB :
���������*
dtype0
�
Qgradients/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/BroadcastGradientArgs/f_acc_1StackV2Qgradients/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/BroadcastGradientArgs/Const_1*
	elem_type0*N
_classD
B@loc:@gradients/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/Shape_1*

stack_name 
�
Qgradients/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/BroadcastGradientArgs/Enter_1EnterQgradients/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *,

frame_namelstm/rnn/while/while_context
�
Wgradients/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2Qgradients/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/BroadcastGradientArgs/Enter_1;gradients/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/Shape_1^gradients/Add*
T0*
swap_memory( 
�
Vgradients/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2\gradients/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/BroadcastGradientArgs/StackPopV2_1/Enter^gradients/Sub*
	elem_type0
�
\gradients/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterQgradients/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *6

frame_name(&gradients/lstm/rnn/while/while_context
�
7gradients/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/MulMulgradients/AddN_5Bgradients/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/StackPopV2*
T0
�
=gradients/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/ConstConst*;
_class1
/-loc:@lstm/rnn/while/basic_lstm_cell/Sigmoid_2*
valueB :
���������*
dtype0
�
=gradients/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/f_accStackV2=gradients/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/Const*
	elem_type0*;
_class1
/-loc:@lstm/rnn/while/basic_lstm_cell/Sigmoid_2*

stack_name 
�
=gradients/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/EnterEnter=gradients/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *,

frame_namelstm/rnn/while/while_context
�
Cgradients/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/StackPushV2StackPushV2=gradients/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/Enter(lstm/rnn/while/basic_lstm_cell/Sigmoid_2^gradients/Add*
T0*
swap_memory( 
�
Bgradients/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/StackPopV2
StackPopV2Hgradients/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/StackPopV2/Enter^gradients/Sub*
	elem_type0
�
Hgradients/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/StackPopV2/EnterEnter=gradients/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *6

frame_name(&gradients/lstm/rnn/while/while_context
�
7gradients/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/SumSum7gradients/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/MulIgradients/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
�
;gradients/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/ReshapeReshape7gradients/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/SumTgradients/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0
�
9gradients/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1MulDgradients/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/StackPopV2gradients/AddN_5*
T0
�
?gradients/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/ConstConst*8
_class.
,*loc:@lstm/rnn/while/basic_lstm_cell/Tanh_1*
valueB :
���������*
dtype0
�
?gradients/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/f_accStackV2?gradients/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/Const*
	elem_type0*8
_class.
,*loc:@lstm/rnn/while/basic_lstm_cell/Tanh_1*

stack_name 
�
?gradients/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/EnterEnter?gradients/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *,

frame_namelstm/rnn/while/while_context
�
Egradients/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/StackPushV2StackPushV2?gradients/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/Enter%lstm/rnn/while/basic_lstm_cell/Tanh_1^gradients/Add*
T0*
swap_memory( 
�
Dgradients/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/StackPopV2
StackPopV2Jgradients/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/StackPopV2/Enter^gradients/Sub*
	elem_type0
�
Jgradients/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/StackPopV2/EnterEnter?gradients/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *6

frame_name(&gradients/lstm/rnn/while/while_context
�
9gradients/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/Sum_1Sum9gradients/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1Kgradients/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
�
=gradients/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/Reshape_1Reshape9gradients/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/Sum_1Vgradients/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/BroadcastGradientArgs/StackPopV2_1*
T0*
Tshape0
�
Dgradients/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/tuple/group_depsNoOp<^gradients/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/Reshape>^gradients/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/Reshape_1
�
Lgradients/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/tuple/control_dependencyIdentity;gradients/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/ReshapeE^gradients/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/tuple/group_deps*
T0*N
_classD
B@loc:@gradients/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/Reshape
�
Ngradients/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/tuple/control_dependency_1Identity=gradients/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/Reshape_1E^gradients/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/tuple/group_deps*
T0*P
_classF
DBloc:@gradients/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/Reshape_1
�
=gradients/lstm/rnn/while/basic_lstm_cell/Tanh_1_grad/TanhGradTanhGradDgradients/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/StackPopV2Lgradients/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/tuple/control_dependency*
T0
�
Cgradients/lstm/rnn/while/basic_lstm_cell/Sigmoid_2_grad/SigmoidGradSigmoidGradBgradients/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/StackPopV2Ngradients/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/tuple/control_dependency_1*
T0
�
6gradients/lstm/rnn/while/Switch_2_grad_1/NextIterationNextIteration\gradients/lstm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency_1*
T0
�
gradients/AddN_6AddN@gradients/lstm/rnn/while/Merge_3_grad/tuple/control_dependency_1=gradients/lstm/rnn/while/basic_lstm_cell/Tanh_1_grad/TanhGrad*
T0*B
_class8
64loc:@gradients/lstm/rnn/while/Switch_3_grad/b_switch*
N

9gradients/lstm/rnn/while/basic_lstm_cell/Add_1_grad/ShapeShape"lstm/rnn/while/basic_lstm_cell/Mul*
T0*
out_type0
�
;gradients/lstm/rnn/while/basic_lstm_cell/Add_1_grad/Shape_1Shape$lstm/rnn/while/basic_lstm_cell/Mul_1*
T0*
out_type0
�
Igradients/lstm/rnn/while/basic_lstm_cell/Add_1_grad/BroadcastGradientArgsBroadcastGradientArgsTgradients/lstm/rnn/while/basic_lstm_cell/Add_1_grad/BroadcastGradientArgs/StackPopV2Vgradients/lstm/rnn/while/basic_lstm_cell/Add_1_grad/BroadcastGradientArgs/StackPopV2_1*
T0
�
Ogradients/lstm/rnn/while/basic_lstm_cell/Add_1_grad/BroadcastGradientArgs/ConstConst*L
_classB
@>loc:@gradients/lstm/rnn/while/basic_lstm_cell/Add_1_grad/Shape*
valueB :
���������*
dtype0
�
Ogradients/lstm/rnn/while/basic_lstm_cell/Add_1_grad/BroadcastGradientArgs/f_accStackV2Ogradients/lstm/rnn/while/basic_lstm_cell/Add_1_grad/BroadcastGradientArgs/Const*
	elem_type0*L
_classB
@>loc:@gradients/lstm/rnn/while/basic_lstm_cell/Add_1_grad/Shape*

stack_name 
�
Ogradients/lstm/rnn/while/basic_lstm_cell/Add_1_grad/BroadcastGradientArgs/EnterEnterOgradients/lstm/rnn/while/basic_lstm_cell/Add_1_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *,

frame_namelstm/rnn/while/while_context
�
Ugradients/lstm/rnn/while/basic_lstm_cell/Add_1_grad/BroadcastGradientArgs/StackPushV2StackPushV2Ogradients/lstm/rnn/while/basic_lstm_cell/Add_1_grad/BroadcastGradientArgs/Enter9gradients/lstm/rnn/while/basic_lstm_cell/Add_1_grad/Shape^gradients/Add*
T0*
swap_memory( 
�
Tgradients/lstm/rnn/while/basic_lstm_cell/Add_1_grad/BroadcastGradientArgs/StackPopV2
StackPopV2Zgradients/lstm/rnn/while/basic_lstm_cell/Add_1_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub*
	elem_type0
�
Zgradients/lstm/rnn/while/basic_lstm_cell/Add_1_grad/BroadcastGradientArgs/StackPopV2/EnterEnterOgradients/lstm/rnn/while/basic_lstm_cell/Add_1_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *6

frame_name(&gradients/lstm/rnn/while/while_context
�
Qgradients/lstm/rnn/while/basic_lstm_cell/Add_1_grad/BroadcastGradientArgs/Const_1Const*N
_classD
B@loc:@gradients/lstm/rnn/while/basic_lstm_cell/Add_1_grad/Shape_1*
valueB :
���������*
dtype0
�
Qgradients/lstm/rnn/while/basic_lstm_cell/Add_1_grad/BroadcastGradientArgs/f_acc_1StackV2Qgradients/lstm/rnn/while/basic_lstm_cell/Add_1_grad/BroadcastGradientArgs/Const_1*
	elem_type0*N
_classD
B@loc:@gradients/lstm/rnn/while/basic_lstm_cell/Add_1_grad/Shape_1*

stack_name 
�
Qgradients/lstm/rnn/while/basic_lstm_cell/Add_1_grad/BroadcastGradientArgs/Enter_1EnterQgradients/lstm/rnn/while/basic_lstm_cell/Add_1_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *,

frame_namelstm/rnn/while/while_context
�
Wgradients/lstm/rnn/while/basic_lstm_cell/Add_1_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2Qgradients/lstm/rnn/while/basic_lstm_cell/Add_1_grad/BroadcastGradientArgs/Enter_1;gradients/lstm/rnn/while/basic_lstm_cell/Add_1_grad/Shape_1^gradients/Add*
T0*
swap_memory( 
�
Vgradients/lstm/rnn/while/basic_lstm_cell/Add_1_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2\gradients/lstm/rnn/while/basic_lstm_cell/Add_1_grad/BroadcastGradientArgs/StackPopV2_1/Enter^gradients/Sub*
	elem_type0
�
\gradients/lstm/rnn/while/basic_lstm_cell/Add_1_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterQgradients/lstm/rnn/while/basic_lstm_cell/Add_1_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *6

frame_name(&gradients/lstm/rnn/while/while_context
�
7gradients/lstm/rnn/while/basic_lstm_cell/Add_1_grad/SumSumgradients/AddN_6Igradients/lstm/rnn/while/basic_lstm_cell/Add_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
�
;gradients/lstm/rnn/while/basic_lstm_cell/Add_1_grad/ReshapeReshape7gradients/lstm/rnn/while/basic_lstm_cell/Add_1_grad/SumTgradients/lstm/rnn/while/basic_lstm_cell/Add_1_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0
�
9gradients/lstm/rnn/while/basic_lstm_cell/Add_1_grad/Sum_1Sumgradients/AddN_6Kgradients/lstm/rnn/while/basic_lstm_cell/Add_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
�
=gradients/lstm/rnn/while/basic_lstm_cell/Add_1_grad/Reshape_1Reshape9gradients/lstm/rnn/while/basic_lstm_cell/Add_1_grad/Sum_1Vgradients/lstm/rnn/while/basic_lstm_cell/Add_1_grad/BroadcastGradientArgs/StackPopV2_1*
T0*
Tshape0
�
Dgradients/lstm/rnn/while/basic_lstm_cell/Add_1_grad/tuple/group_depsNoOp<^gradients/lstm/rnn/while/basic_lstm_cell/Add_1_grad/Reshape>^gradients/lstm/rnn/while/basic_lstm_cell/Add_1_grad/Reshape_1
�
Lgradients/lstm/rnn/while/basic_lstm_cell/Add_1_grad/tuple/control_dependencyIdentity;gradients/lstm/rnn/while/basic_lstm_cell/Add_1_grad/ReshapeE^gradients/lstm/rnn/while/basic_lstm_cell/Add_1_grad/tuple/group_deps*
T0*N
_classD
B@loc:@gradients/lstm/rnn/while/basic_lstm_cell/Add_1_grad/Reshape
�
Ngradients/lstm/rnn/while/basic_lstm_cell/Add_1_grad/tuple/control_dependency_1Identity=gradients/lstm/rnn/while/basic_lstm_cell/Add_1_grad/Reshape_1E^gradients/lstm/rnn/while/basic_lstm_cell/Add_1_grad/tuple/group_deps*
T0*P
_classF
DBloc:@gradients/lstm/rnn/while/basic_lstm_cell/Add_1_grad/Reshape_1
t
7gradients/lstm/rnn/while/basic_lstm_cell/Mul_grad/ShapeShapelstm/rnn/while/Identity_3*
T0*
out_type0
�
9gradients/lstm/rnn/while/basic_lstm_cell/Mul_grad/Shape_1Shape&lstm/rnn/while/basic_lstm_cell/Sigmoid*
T0*
out_type0
�
Ggradients/lstm/rnn/while/basic_lstm_cell/Mul_grad/BroadcastGradientArgsBroadcastGradientArgsRgradients/lstm/rnn/while/basic_lstm_cell/Mul_grad/BroadcastGradientArgs/StackPopV2Tgradients/lstm/rnn/while/basic_lstm_cell/Mul_grad/BroadcastGradientArgs/StackPopV2_1*
T0
�
Mgradients/lstm/rnn/while/basic_lstm_cell/Mul_grad/BroadcastGradientArgs/ConstConst*J
_class@
><loc:@gradients/lstm/rnn/while/basic_lstm_cell/Mul_grad/Shape*
valueB :
���������*
dtype0
�
Mgradients/lstm/rnn/while/basic_lstm_cell/Mul_grad/BroadcastGradientArgs/f_accStackV2Mgradients/lstm/rnn/while/basic_lstm_cell/Mul_grad/BroadcastGradientArgs/Const*
	elem_type0*J
_class@
><loc:@gradients/lstm/rnn/while/basic_lstm_cell/Mul_grad/Shape*

stack_name 
�
Mgradients/lstm/rnn/while/basic_lstm_cell/Mul_grad/BroadcastGradientArgs/EnterEnterMgradients/lstm/rnn/while/basic_lstm_cell/Mul_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *,

frame_namelstm/rnn/while/while_context
�
Sgradients/lstm/rnn/while/basic_lstm_cell/Mul_grad/BroadcastGradientArgs/StackPushV2StackPushV2Mgradients/lstm/rnn/while/basic_lstm_cell/Mul_grad/BroadcastGradientArgs/Enter7gradients/lstm/rnn/while/basic_lstm_cell/Mul_grad/Shape^gradients/Add*
T0*
swap_memory( 
�
Rgradients/lstm/rnn/while/basic_lstm_cell/Mul_grad/BroadcastGradientArgs/StackPopV2
StackPopV2Xgradients/lstm/rnn/while/basic_lstm_cell/Mul_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub*
	elem_type0
�
Xgradients/lstm/rnn/while/basic_lstm_cell/Mul_grad/BroadcastGradientArgs/StackPopV2/EnterEnterMgradients/lstm/rnn/while/basic_lstm_cell/Mul_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *6

frame_name(&gradients/lstm/rnn/while/while_context
�
Ogradients/lstm/rnn/while/basic_lstm_cell/Mul_grad/BroadcastGradientArgs/Const_1Const*L
_classB
@>loc:@gradients/lstm/rnn/while/basic_lstm_cell/Mul_grad/Shape_1*
valueB :
���������*
dtype0
�
Ogradients/lstm/rnn/while/basic_lstm_cell/Mul_grad/BroadcastGradientArgs/f_acc_1StackV2Ogradients/lstm/rnn/while/basic_lstm_cell/Mul_grad/BroadcastGradientArgs/Const_1*
	elem_type0*L
_classB
@>loc:@gradients/lstm/rnn/while/basic_lstm_cell/Mul_grad/Shape_1*

stack_name 
�
Ogradients/lstm/rnn/while/basic_lstm_cell/Mul_grad/BroadcastGradientArgs/Enter_1EnterOgradients/lstm/rnn/while/basic_lstm_cell/Mul_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *,

frame_namelstm/rnn/while/while_context
�
Ugradients/lstm/rnn/while/basic_lstm_cell/Mul_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2Ogradients/lstm/rnn/while/basic_lstm_cell/Mul_grad/BroadcastGradientArgs/Enter_19gradients/lstm/rnn/while/basic_lstm_cell/Mul_grad/Shape_1^gradients/Add*
T0*
swap_memory( 
�
Tgradients/lstm/rnn/while/basic_lstm_cell/Mul_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2Zgradients/lstm/rnn/while/basic_lstm_cell/Mul_grad/BroadcastGradientArgs/StackPopV2_1/Enter^gradients/Sub*
	elem_type0
�
Zgradients/lstm/rnn/while/basic_lstm_cell/Mul_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterOgradients/lstm/rnn/while/basic_lstm_cell/Mul_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *6

frame_name(&gradients/lstm/rnn/while/while_context
�
5gradients/lstm/rnn/while/basic_lstm_cell/Mul_grad/MulMulLgradients/lstm/rnn/while/basic_lstm_cell/Add_1_grad/tuple/control_dependency@gradients/lstm/rnn/while/basic_lstm_cell/Mul_grad/Mul/StackPopV2*
T0
�
;gradients/lstm/rnn/while/basic_lstm_cell/Mul_grad/Mul/ConstConst*9
_class/
-+loc:@lstm/rnn/while/basic_lstm_cell/Sigmoid*
valueB :
���������*
dtype0
�
;gradients/lstm/rnn/while/basic_lstm_cell/Mul_grad/Mul/f_accStackV2;gradients/lstm/rnn/while/basic_lstm_cell/Mul_grad/Mul/Const*
	elem_type0*9
_class/
-+loc:@lstm/rnn/while/basic_lstm_cell/Sigmoid*

stack_name 
�
;gradients/lstm/rnn/while/basic_lstm_cell/Mul_grad/Mul/EnterEnter;gradients/lstm/rnn/while/basic_lstm_cell/Mul_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *,

frame_namelstm/rnn/while/while_context
�
Agradients/lstm/rnn/while/basic_lstm_cell/Mul_grad/Mul/StackPushV2StackPushV2;gradients/lstm/rnn/while/basic_lstm_cell/Mul_grad/Mul/Enter&lstm/rnn/while/basic_lstm_cell/Sigmoid^gradients/Add*
T0*
swap_memory( 
�
@gradients/lstm/rnn/while/basic_lstm_cell/Mul_grad/Mul/StackPopV2
StackPopV2Fgradients/lstm/rnn/while/basic_lstm_cell/Mul_grad/Mul/StackPopV2/Enter^gradients/Sub*
	elem_type0
�
Fgradients/lstm/rnn/while/basic_lstm_cell/Mul_grad/Mul/StackPopV2/EnterEnter;gradients/lstm/rnn/while/basic_lstm_cell/Mul_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *6

frame_name(&gradients/lstm/rnn/while/while_context
�
5gradients/lstm/rnn/while/basic_lstm_cell/Mul_grad/SumSum5gradients/lstm/rnn/while/basic_lstm_cell/Mul_grad/MulGgradients/lstm/rnn/while/basic_lstm_cell/Mul_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
�
9gradients/lstm/rnn/while/basic_lstm_cell/Mul_grad/ReshapeReshape5gradients/lstm/rnn/while/basic_lstm_cell/Mul_grad/SumRgradients/lstm/rnn/while/basic_lstm_cell/Mul_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0
�
7gradients/lstm/rnn/while/basic_lstm_cell/Mul_grad/Mul_1MulBgradients/lstm/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/StackPopV2Lgradients/lstm/rnn/while/basic_lstm_cell/Add_1_grad/tuple/control_dependency*
T0
�
=gradients/lstm/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/ConstConst*,
_class"
 loc:@lstm/rnn/while/Identity_3*
valueB :
���������*
dtype0
�
=gradients/lstm/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/f_accStackV2=gradients/lstm/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/Const*
	elem_type0*,
_class"
 loc:@lstm/rnn/while/Identity_3*

stack_name 
�
=gradients/lstm/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/EnterEnter=gradients/lstm/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *,

frame_namelstm/rnn/while/while_context
�
Cgradients/lstm/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/StackPushV2StackPushV2=gradients/lstm/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/Enterlstm/rnn/while/Identity_3^gradients/Add*
T0*
swap_memory( 
�
Bgradients/lstm/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/StackPopV2
StackPopV2Hgradients/lstm/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/StackPopV2/Enter^gradients/Sub*
	elem_type0
�
Hgradients/lstm/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/StackPopV2/EnterEnter=gradients/lstm/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *6

frame_name(&gradients/lstm/rnn/while/while_context
�
7gradients/lstm/rnn/while/basic_lstm_cell/Mul_grad/Sum_1Sum7gradients/lstm/rnn/while/basic_lstm_cell/Mul_grad/Mul_1Igradients/lstm/rnn/while/basic_lstm_cell/Mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
�
;gradients/lstm/rnn/while/basic_lstm_cell/Mul_grad/Reshape_1Reshape7gradients/lstm/rnn/while/basic_lstm_cell/Mul_grad/Sum_1Tgradients/lstm/rnn/while/basic_lstm_cell/Mul_grad/BroadcastGradientArgs/StackPopV2_1*
T0*
Tshape0
�
Bgradients/lstm/rnn/while/basic_lstm_cell/Mul_grad/tuple/group_depsNoOp:^gradients/lstm/rnn/while/basic_lstm_cell/Mul_grad/Reshape<^gradients/lstm/rnn/while/basic_lstm_cell/Mul_grad/Reshape_1
�
Jgradients/lstm/rnn/while/basic_lstm_cell/Mul_grad/tuple/control_dependencyIdentity9gradients/lstm/rnn/while/basic_lstm_cell/Mul_grad/ReshapeC^gradients/lstm/rnn/while/basic_lstm_cell/Mul_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients/lstm/rnn/while/basic_lstm_cell/Mul_grad/Reshape
�
Lgradients/lstm/rnn/while/basic_lstm_cell/Mul_grad/tuple/control_dependency_1Identity;gradients/lstm/rnn/while/basic_lstm_cell/Mul_grad/Reshape_1C^gradients/lstm/rnn/while/basic_lstm_cell/Mul_grad/tuple/group_deps*
T0*N
_classD
B@loc:@gradients/lstm/rnn/while/basic_lstm_cell/Mul_grad/Reshape_1
�
9gradients/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/ShapeShape(lstm/rnn/while/basic_lstm_cell/Sigmoid_1*
T0*
out_type0
�
;gradients/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/Shape_1Shape#lstm/rnn/while/basic_lstm_cell/Tanh*
T0*
out_type0
�
Igradients/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsTgradients/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/BroadcastGradientArgs/StackPopV2Vgradients/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/BroadcastGradientArgs/StackPopV2_1*
T0
�
Ogradients/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/BroadcastGradientArgs/ConstConst*L
_classB
@>loc:@gradients/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/Shape*
valueB :
���������*
dtype0
�
Ogradients/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/BroadcastGradientArgs/f_accStackV2Ogradients/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/BroadcastGradientArgs/Const*
	elem_type0*L
_classB
@>loc:@gradients/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/Shape*

stack_name 
�
Ogradients/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/BroadcastGradientArgs/EnterEnterOgradients/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *,

frame_namelstm/rnn/while/while_context
�
Ugradients/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/BroadcastGradientArgs/StackPushV2StackPushV2Ogradients/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/BroadcastGradientArgs/Enter9gradients/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/Shape^gradients/Add*
T0*
swap_memory( 
�
Tgradients/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/BroadcastGradientArgs/StackPopV2
StackPopV2Zgradients/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub*
	elem_type0
�
Zgradients/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/BroadcastGradientArgs/StackPopV2/EnterEnterOgradients/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *6

frame_name(&gradients/lstm/rnn/while/while_context
�
Qgradients/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/BroadcastGradientArgs/Const_1Const*N
_classD
B@loc:@gradients/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/Shape_1*
valueB :
���������*
dtype0
�
Qgradients/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/BroadcastGradientArgs/f_acc_1StackV2Qgradients/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/BroadcastGradientArgs/Const_1*
	elem_type0*N
_classD
B@loc:@gradients/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/Shape_1*

stack_name 
�
Qgradients/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/BroadcastGradientArgs/Enter_1EnterQgradients/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *,

frame_namelstm/rnn/while/while_context
�
Wgradients/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2Qgradients/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/BroadcastGradientArgs/Enter_1;gradients/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/Shape_1^gradients/Add*
T0*
swap_memory( 
�
Vgradients/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2\gradients/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/BroadcastGradientArgs/StackPopV2_1/Enter^gradients/Sub*
	elem_type0
�
\gradients/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterQgradients/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *6

frame_name(&gradients/lstm/rnn/while/while_context
�
7gradients/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/MulMulNgradients/lstm/rnn/while/basic_lstm_cell/Add_1_grad/tuple/control_dependency_1Bgradients/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/StackPopV2*
T0
�
=gradients/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/ConstConst*6
_class,
*(loc:@lstm/rnn/while/basic_lstm_cell/Tanh*
valueB :
���������*
dtype0
�
=gradients/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/f_accStackV2=gradients/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/Const*
	elem_type0*6
_class,
*(loc:@lstm/rnn/while/basic_lstm_cell/Tanh*

stack_name 
�
=gradients/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/EnterEnter=gradients/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *,

frame_namelstm/rnn/while/while_context
�
Cgradients/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/StackPushV2StackPushV2=gradients/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/Enter#lstm/rnn/while/basic_lstm_cell/Tanh^gradients/Add*
T0*
swap_memory( 
�
Bgradients/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/StackPopV2
StackPopV2Hgradients/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/StackPopV2/Enter^gradients/Sub*
	elem_type0
�
Hgradients/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/StackPopV2/EnterEnter=gradients/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *6

frame_name(&gradients/lstm/rnn/while/while_context
�
7gradients/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/SumSum7gradients/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/MulIgradients/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
�
;gradients/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/ReshapeReshape7gradients/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/SumTgradients/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0
�
9gradients/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1MulDgradients/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/StackPopV2Ngradients/lstm/rnn/while/basic_lstm_cell/Add_1_grad/tuple/control_dependency_1*
T0
�
?gradients/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/ConstConst*;
_class1
/-loc:@lstm/rnn/while/basic_lstm_cell/Sigmoid_1*
valueB :
���������*
dtype0
�
?gradients/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/f_accStackV2?gradients/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/Const*
	elem_type0*;
_class1
/-loc:@lstm/rnn/while/basic_lstm_cell/Sigmoid_1*

stack_name 
�
?gradients/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/EnterEnter?gradients/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *,

frame_namelstm/rnn/while/while_context
�
Egradients/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/StackPushV2StackPushV2?gradients/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/Enter(lstm/rnn/while/basic_lstm_cell/Sigmoid_1^gradients/Add*
T0*
swap_memory( 
�
Dgradients/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/StackPopV2
StackPopV2Jgradients/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/StackPopV2/Enter^gradients/Sub*
	elem_type0
�
Jgradients/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/StackPopV2/EnterEnter?gradients/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *6

frame_name(&gradients/lstm/rnn/while/while_context
�
9gradients/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/Sum_1Sum9gradients/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1Kgradients/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
�
=gradients/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/Reshape_1Reshape9gradients/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/Sum_1Vgradients/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/BroadcastGradientArgs/StackPopV2_1*
T0*
Tshape0
�
Dgradients/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/tuple/group_depsNoOp<^gradients/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/Reshape>^gradients/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/Reshape_1
�
Lgradients/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/tuple/control_dependencyIdentity;gradients/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/ReshapeE^gradients/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/tuple/group_deps*
T0*N
_classD
B@loc:@gradients/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/Reshape
�
Ngradients/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/tuple/control_dependency_1Identity=gradients/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/Reshape_1E^gradients/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/tuple/group_deps*
T0*P
_classF
DBloc:@gradients/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/Reshape_1
�
Agradients/lstm/rnn/while/basic_lstm_cell/Sigmoid_grad/SigmoidGradSigmoidGrad@gradients/lstm/rnn/while/basic_lstm_cell/Mul_grad/Mul/StackPopV2Lgradients/lstm/rnn/while/basic_lstm_cell/Mul_grad/tuple/control_dependency_1*
T0
�
Cgradients/lstm/rnn/while/basic_lstm_cell/Sigmoid_1_grad/SigmoidGradSigmoidGradDgradients/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/StackPopV2Lgradients/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/tuple/control_dependency*
T0
�
;gradients/lstm/rnn/while/basic_lstm_cell/Tanh_grad/TanhGradTanhGradBgradients/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/StackPopV2Ngradients/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/tuple/control_dependency_1*
T0
�
6gradients/lstm/rnn/while/Switch_3_grad_1/NextIterationNextIterationJgradients/lstm/rnn/while/basic_lstm_cell/Mul_grad/tuple/control_dependency*
T0
�
7gradients/lstm/rnn/while/basic_lstm_cell/Add_grad/ShapeShape&lstm/rnn/while/basic_lstm_cell/split:2*
T0*
out_type0
�
9gradients/lstm/rnn/while/basic_lstm_cell/Add_grad/Shape_1Shape&lstm/rnn/while/basic_lstm_cell/Const_2*
T0*
out_type0
�
Ggradients/lstm/rnn/while/basic_lstm_cell/Add_grad/BroadcastGradientArgsBroadcastGradientArgsRgradients/lstm/rnn/while/basic_lstm_cell/Add_grad/BroadcastGradientArgs/StackPopV2Tgradients/lstm/rnn/while/basic_lstm_cell/Add_grad/BroadcastGradientArgs/StackPopV2_1*
T0
�
Mgradients/lstm/rnn/while/basic_lstm_cell/Add_grad/BroadcastGradientArgs/ConstConst*J
_class@
><loc:@gradients/lstm/rnn/while/basic_lstm_cell/Add_grad/Shape*
valueB :
���������*
dtype0
�
Mgradients/lstm/rnn/while/basic_lstm_cell/Add_grad/BroadcastGradientArgs/f_accStackV2Mgradients/lstm/rnn/while/basic_lstm_cell/Add_grad/BroadcastGradientArgs/Const*
	elem_type0*J
_class@
><loc:@gradients/lstm/rnn/while/basic_lstm_cell/Add_grad/Shape*

stack_name 
�
Mgradients/lstm/rnn/while/basic_lstm_cell/Add_grad/BroadcastGradientArgs/EnterEnterMgradients/lstm/rnn/while/basic_lstm_cell/Add_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *,

frame_namelstm/rnn/while/while_context
�
Sgradients/lstm/rnn/while/basic_lstm_cell/Add_grad/BroadcastGradientArgs/StackPushV2StackPushV2Mgradients/lstm/rnn/while/basic_lstm_cell/Add_grad/BroadcastGradientArgs/Enter7gradients/lstm/rnn/while/basic_lstm_cell/Add_grad/Shape^gradients/Add*
T0*
swap_memory( 
�
Rgradients/lstm/rnn/while/basic_lstm_cell/Add_grad/BroadcastGradientArgs/StackPopV2
StackPopV2Xgradients/lstm/rnn/while/basic_lstm_cell/Add_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients/Sub*
	elem_type0
�
Xgradients/lstm/rnn/while/basic_lstm_cell/Add_grad/BroadcastGradientArgs/StackPopV2/EnterEnterMgradients/lstm/rnn/while/basic_lstm_cell/Add_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *6

frame_name(&gradients/lstm/rnn/while/while_context
�
Ogradients/lstm/rnn/while/basic_lstm_cell/Add_grad/BroadcastGradientArgs/Const_1Const*L
_classB
@>loc:@gradients/lstm/rnn/while/basic_lstm_cell/Add_grad/Shape_1*
valueB :
���������*
dtype0
�
Ogradients/lstm/rnn/while/basic_lstm_cell/Add_grad/BroadcastGradientArgs/f_acc_1StackV2Ogradients/lstm/rnn/while/basic_lstm_cell/Add_grad/BroadcastGradientArgs/Const_1*
	elem_type0*L
_classB
@>loc:@gradients/lstm/rnn/while/basic_lstm_cell/Add_grad/Shape_1*

stack_name 
�
Ogradients/lstm/rnn/while/basic_lstm_cell/Add_grad/BroadcastGradientArgs/Enter_1EnterOgradients/lstm/rnn/while/basic_lstm_cell/Add_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *,

frame_namelstm/rnn/while/while_context
�
Ugradients/lstm/rnn/while/basic_lstm_cell/Add_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2Ogradients/lstm/rnn/while/basic_lstm_cell/Add_grad/BroadcastGradientArgs/Enter_19gradients/lstm/rnn/while/basic_lstm_cell/Add_grad/Shape_1^gradients/Add*
T0*
swap_memory( 
�
Tgradients/lstm/rnn/while/basic_lstm_cell/Add_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2Zgradients/lstm/rnn/while/basic_lstm_cell/Add_grad/BroadcastGradientArgs/StackPopV2_1/Enter^gradients/Sub*
	elem_type0
�
Zgradients/lstm/rnn/while/basic_lstm_cell/Add_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterOgradients/lstm/rnn/while/basic_lstm_cell/Add_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *6

frame_name(&gradients/lstm/rnn/while/while_context
�
5gradients/lstm/rnn/while/basic_lstm_cell/Add_grad/SumSumAgradients/lstm/rnn/while/basic_lstm_cell/Sigmoid_grad/SigmoidGradGgradients/lstm/rnn/while/basic_lstm_cell/Add_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
�
9gradients/lstm/rnn/while/basic_lstm_cell/Add_grad/ReshapeReshape5gradients/lstm/rnn/while/basic_lstm_cell/Add_grad/SumRgradients/lstm/rnn/while/basic_lstm_cell/Add_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0
�
7gradients/lstm/rnn/while/basic_lstm_cell/Add_grad/Sum_1SumAgradients/lstm/rnn/while/basic_lstm_cell/Sigmoid_grad/SigmoidGradIgradients/lstm/rnn/while/basic_lstm_cell/Add_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
�
;gradients/lstm/rnn/while/basic_lstm_cell/Add_grad/Reshape_1Reshape7gradients/lstm/rnn/while/basic_lstm_cell/Add_grad/Sum_1Tgradients/lstm/rnn/while/basic_lstm_cell/Add_grad/BroadcastGradientArgs/StackPopV2_1*
T0*
Tshape0
�
Bgradients/lstm/rnn/while/basic_lstm_cell/Add_grad/tuple/group_depsNoOp:^gradients/lstm/rnn/while/basic_lstm_cell/Add_grad/Reshape<^gradients/lstm/rnn/while/basic_lstm_cell/Add_grad/Reshape_1
�
Jgradients/lstm/rnn/while/basic_lstm_cell/Add_grad/tuple/control_dependencyIdentity9gradients/lstm/rnn/while/basic_lstm_cell/Add_grad/ReshapeC^gradients/lstm/rnn/while/basic_lstm_cell/Add_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients/lstm/rnn/while/basic_lstm_cell/Add_grad/Reshape
�
Lgradients/lstm/rnn/while/basic_lstm_cell/Add_grad/tuple/control_dependency_1Identity;gradients/lstm/rnn/while/basic_lstm_cell/Add_grad/Reshape_1C^gradients/lstm/rnn/while/basic_lstm_cell/Add_grad/tuple/group_deps*
T0*N
_classD
B@loc:@gradients/lstm/rnn/while/basic_lstm_cell/Add_grad/Reshape_1
�
:gradients/lstm/rnn/while/basic_lstm_cell/split_grad/concatConcatV2Cgradients/lstm/rnn/while/basic_lstm_cell/Sigmoid_1_grad/SigmoidGrad;gradients/lstm/rnn/while/basic_lstm_cell/Tanh_grad/TanhGradJgradients/lstm/rnn/while/basic_lstm_cell/Add_grad/tuple/control_dependencyCgradients/lstm/rnn/while/basic_lstm_cell/Sigmoid_2_grad/SigmoidGrad@gradients/lstm/rnn/while/basic_lstm_cell/split_grad/concat/Const*

Tidx0*
T0*
N
z
@gradients/lstm/rnn/while/basic_lstm_cell/split_grad/concat/ConstConst^gradients/Sub*
value	B :*
dtype0
�
Agradients/lstm/rnn/while/basic_lstm_cell/BiasAdd_grad/BiasAddGradBiasAddGrad:gradients/lstm/rnn/while/basic_lstm_cell/split_grad/concat*
T0*
data_formatNHWC
�
Fgradients/lstm/rnn/while/basic_lstm_cell/BiasAdd_grad/tuple/group_depsNoOpB^gradients/lstm/rnn/while/basic_lstm_cell/BiasAdd_grad/BiasAddGrad;^gradients/lstm/rnn/while/basic_lstm_cell/split_grad/concat
�
Ngradients/lstm/rnn/while/basic_lstm_cell/BiasAdd_grad/tuple/control_dependencyIdentity:gradients/lstm/rnn/while/basic_lstm_cell/split_grad/concatG^gradients/lstm/rnn/while/basic_lstm_cell/BiasAdd_grad/tuple/group_deps*
T0*M
_classC
A?loc:@gradients/lstm/rnn/while/basic_lstm_cell/split_grad/concat
�
Pgradients/lstm/rnn/while/basic_lstm_cell/BiasAdd_grad/tuple/control_dependency_1IdentityAgradients/lstm/rnn/while/basic_lstm_cell/BiasAdd_grad/BiasAddGradG^gradients/lstm/rnn/while/basic_lstm_cell/BiasAdd_grad/tuple/group_deps*
T0*T
_classJ
HFloc:@gradients/lstm/rnn/while/basic_lstm_cell/BiasAdd_grad/BiasAddGrad
�
;gradients/lstm/rnn/while/basic_lstm_cell/MatMul_grad/MatMulMatMulNgradients/lstm/rnn/while/basic_lstm_cell/BiasAdd_grad/tuple/control_dependencyAgradients/lstm/rnn/while/basic_lstm_cell/MatMul_grad/MatMul/Enter*
transpose_b(*
T0*
transpose_a( 
�
Agradients/lstm/rnn/while/basic_lstm_cell/MatMul_grad/MatMul/EnterEnter$lstm/rnn/basic_lstm_cell/kernel/read*
T0*
is_constant(*
parallel_iterations *6

frame_name(&gradients/lstm/rnn/while/while_context
�
=gradients/lstm/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1MatMulHgradients/lstm/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2Ngradients/lstm/rnn/while/basic_lstm_cell/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(
�
Cgradients/lstm/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/ConstConst*8
_class.
,*loc:@lstm/rnn/while/basic_lstm_cell/concat*
valueB :
���������*
dtype0
�
Cgradients/lstm/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/f_accStackV2Cgradients/lstm/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/Const*
	elem_type0*8
_class.
,*loc:@lstm/rnn/while/basic_lstm_cell/concat*

stack_name 
�
Cgradients/lstm/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/EnterEnterCgradients/lstm/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/f_acc*
T0*
is_constant(*
parallel_iterations *,

frame_namelstm/rnn/while/while_context
�
Igradients/lstm/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/StackPushV2StackPushV2Cgradients/lstm/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/Enter%lstm/rnn/while/basic_lstm_cell/concat^gradients/Add*
T0*
swap_memory( 
�
Hgradients/lstm/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2
StackPopV2Ngradients/lstm/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2/Enter^gradients/Sub*
	elem_type0
�
Ngradients/lstm/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2/EnterEnterCgradients/lstm/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/f_acc*
T0*
is_constant(*
parallel_iterations *6

frame_name(&gradients/lstm/rnn/while/while_context
�
Egradients/lstm/rnn/while/basic_lstm_cell/MatMul_grad/tuple/group_depsNoOp<^gradients/lstm/rnn/while/basic_lstm_cell/MatMul_grad/MatMul>^gradients/lstm/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1
�
Mgradients/lstm/rnn/while/basic_lstm_cell/MatMul_grad/tuple/control_dependencyIdentity;gradients/lstm/rnn/while/basic_lstm_cell/MatMul_grad/MatMulF^gradients/lstm/rnn/while/basic_lstm_cell/MatMul_grad/tuple/group_deps*
T0*N
_classD
B@loc:@gradients/lstm/rnn/while/basic_lstm_cell/MatMul_grad/MatMul
�
Ogradients/lstm/rnn/while/basic_lstm_cell/MatMul_grad/tuple/control_dependency_1Identity=gradients/lstm/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1F^gradients/lstm/rnn/while/basic_lstm_cell/MatMul_grad/tuple/group_deps*
T0*P
_classF
DBloc:@gradients/lstm/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1
s
Agradients/lstm/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/b_accConst*
valueB�*    *
dtype0
�
Cgradients/lstm/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_1EnterAgradients/lstm/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/b_acc*
T0*
is_constant( *
parallel_iterations *6

frame_name(&gradients/lstm/rnn/while/while_context
�
Cgradients/lstm/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_2MergeCgradients/lstm/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_1Igradients/lstm/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/NextIteration*
T0*
N
�
Bgradients/lstm/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/SwitchSwitchCgradients/lstm/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_2gradients/b_count_2*
T0
�
?gradients/lstm/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/AddAddDgradients/lstm/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/Switch:1Pgradients/lstm/rnn/while/basic_lstm_cell/BiasAdd_grad/tuple/control_dependency_1*
T0
�
Igradients/lstm/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/NextIterationNextIteration?gradients/lstm/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/Add*
T0
�
Cgradients/lstm/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_3ExitBgradients/lstm/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/Switch*
T0
t
:gradients/lstm/rnn/while/basic_lstm_cell/concat_grad/ConstConst^gradients/Sub*
value	B :*
dtype0
s
9gradients/lstm/rnn/while/basic_lstm_cell/concat_grad/RankConst^gradients/Sub*
value	B :*
dtype0
�
8gradients/lstm/rnn/while/basic_lstm_cell/concat_grad/modFloorMod:gradients/lstm/rnn/while/basic_lstm_cell/concat_grad/Const9gradients/lstm/rnn/while/basic_lstm_cell/concat_grad/Rank*
T0
~
:gradients/lstm/rnn/while/basic_lstm_cell/concat_grad/ShapeShape lstm/rnn/while/TensorArrayReadV3*
T0*
out_type0
�
;gradients/lstm/rnn/while/basic_lstm_cell/concat_grad/ShapeNShapeNFgradients/lstm/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPopV2Hgradients/lstm/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPopV2_1*
T0*
out_type0*
N
�
Agradients/lstm/rnn/while/basic_lstm_cell/concat_grad/ShapeN/ConstConst*3
_class)
'%loc:@lstm/rnn/while/TensorArrayReadV3*
valueB :
���������*
dtype0
�
Agradients/lstm/rnn/while/basic_lstm_cell/concat_grad/ShapeN/f_accStackV2Agradients/lstm/rnn/while/basic_lstm_cell/concat_grad/ShapeN/Const*
	elem_type0*3
_class)
'%loc:@lstm/rnn/while/TensorArrayReadV3*

stack_name 
�
Agradients/lstm/rnn/while/basic_lstm_cell/concat_grad/ShapeN/EnterEnterAgradients/lstm/rnn/while/basic_lstm_cell/concat_grad/ShapeN/f_acc*
T0*
is_constant(*
parallel_iterations *,

frame_namelstm/rnn/while/while_context
�
Ggradients/lstm/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPushV2StackPushV2Agradients/lstm/rnn/while/basic_lstm_cell/concat_grad/ShapeN/Enter lstm/rnn/while/TensorArrayReadV3^gradients/Add*
T0*
swap_memory( 
�
Fgradients/lstm/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPopV2
StackPopV2Lgradients/lstm/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPopV2/Enter^gradients/Sub*
	elem_type0
�
Lgradients/lstm/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPopV2/EnterEnterAgradients/lstm/rnn/while/basic_lstm_cell/concat_grad/ShapeN/f_acc*
T0*
is_constant(*
parallel_iterations *6

frame_name(&gradients/lstm/rnn/while/while_context
�
Cgradients/lstm/rnn/while/basic_lstm_cell/concat_grad/ShapeN/Const_1Const*,
_class"
 loc:@lstm/rnn/while/Identity_4*
valueB :
���������*
dtype0
�
Cgradients/lstm/rnn/while/basic_lstm_cell/concat_grad/ShapeN/f_acc_1StackV2Cgradients/lstm/rnn/while/basic_lstm_cell/concat_grad/ShapeN/Const_1*
	elem_type0*,
_class"
 loc:@lstm/rnn/while/Identity_4*

stack_name 
�
Cgradients/lstm/rnn/while/basic_lstm_cell/concat_grad/ShapeN/Enter_1EnterCgradients/lstm/rnn/while/basic_lstm_cell/concat_grad/ShapeN/f_acc_1*
T0*
is_constant(*
parallel_iterations *,

frame_namelstm/rnn/while/while_context
�
Igradients/lstm/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPushV2_1StackPushV2Cgradients/lstm/rnn/while/basic_lstm_cell/concat_grad/ShapeN/Enter_1lstm/rnn/while/Identity_4^gradients/Add*
T0*
swap_memory( 
�
Hgradients/lstm/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPopV2_1
StackPopV2Ngradients/lstm/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPopV2_1/Enter^gradients/Sub*
	elem_type0
�
Ngradients/lstm/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPopV2_1/EnterEnterCgradients/lstm/rnn/while/basic_lstm_cell/concat_grad/ShapeN/f_acc_1*
T0*
is_constant(*
parallel_iterations *6

frame_name(&gradients/lstm/rnn/while/while_context
�
Agradients/lstm/rnn/while/basic_lstm_cell/concat_grad/ConcatOffsetConcatOffset8gradients/lstm/rnn/while/basic_lstm_cell/concat_grad/mod;gradients/lstm/rnn/while/basic_lstm_cell/concat_grad/ShapeN=gradients/lstm/rnn/while/basic_lstm_cell/concat_grad/ShapeN:1*
N
�
:gradients/lstm/rnn/while/basic_lstm_cell/concat_grad/SliceSliceMgradients/lstm/rnn/while/basic_lstm_cell/MatMul_grad/tuple/control_dependencyAgradients/lstm/rnn/while/basic_lstm_cell/concat_grad/ConcatOffset;gradients/lstm/rnn/while/basic_lstm_cell/concat_grad/ShapeN*
T0*
Index0
�
<gradients/lstm/rnn/while/basic_lstm_cell/concat_grad/Slice_1SliceMgradients/lstm/rnn/while/basic_lstm_cell/MatMul_grad/tuple/control_dependencyCgradients/lstm/rnn/while/basic_lstm_cell/concat_grad/ConcatOffset:1=gradients/lstm/rnn/while/basic_lstm_cell/concat_grad/ShapeN:1*
T0*
Index0
�
Egradients/lstm/rnn/while/basic_lstm_cell/concat_grad/tuple/group_depsNoOp;^gradients/lstm/rnn/while/basic_lstm_cell/concat_grad/Slice=^gradients/lstm/rnn/while/basic_lstm_cell/concat_grad/Slice_1
�
Mgradients/lstm/rnn/while/basic_lstm_cell/concat_grad/tuple/control_dependencyIdentity:gradients/lstm/rnn/while/basic_lstm_cell/concat_grad/SliceF^gradients/lstm/rnn/while/basic_lstm_cell/concat_grad/tuple/group_deps*
T0*M
_classC
A?loc:@gradients/lstm/rnn/while/basic_lstm_cell/concat_grad/Slice
�
Ogradients/lstm/rnn/while/basic_lstm_cell/concat_grad/tuple/control_dependency_1Identity<gradients/lstm/rnn/while/basic_lstm_cell/concat_grad/Slice_1F^gradients/lstm/rnn/while/basic_lstm_cell/concat_grad/tuple/group_deps*
T0*O
_classE
CAloc:@gradients/lstm/rnn/while/basic_lstm_cell/concat_grad/Slice_1
w
@gradients/lstm/rnn/while/basic_lstm_cell/MatMul/Enter_grad/b_accConst*
valueB
��*    *
dtype0
�
Bgradients/lstm/rnn/while/basic_lstm_cell/MatMul/Enter_grad/b_acc_1Enter@gradients/lstm/rnn/while/basic_lstm_cell/MatMul/Enter_grad/b_acc*
T0*
is_constant( *
parallel_iterations *6

frame_name(&gradients/lstm/rnn/while/while_context
�
Bgradients/lstm/rnn/while/basic_lstm_cell/MatMul/Enter_grad/b_acc_2MergeBgradients/lstm/rnn/while/basic_lstm_cell/MatMul/Enter_grad/b_acc_1Hgradients/lstm/rnn/while/basic_lstm_cell/MatMul/Enter_grad/NextIteration*
T0*
N
�
Agradients/lstm/rnn/while/basic_lstm_cell/MatMul/Enter_grad/SwitchSwitchBgradients/lstm/rnn/while/basic_lstm_cell/MatMul/Enter_grad/b_acc_2gradients/b_count_2*
T0
�
>gradients/lstm/rnn/while/basic_lstm_cell/MatMul/Enter_grad/AddAddCgradients/lstm/rnn/while/basic_lstm_cell/MatMul/Enter_grad/Switch:1Ogradients/lstm/rnn/while/basic_lstm_cell/MatMul_grad/tuple/control_dependency_1*
T0
�
Hgradients/lstm/rnn/while/basic_lstm_cell/MatMul/Enter_grad/NextIterationNextIteration>gradients/lstm/rnn/while/basic_lstm_cell/MatMul/Enter_grad/Add*
T0
�
Bgradients/lstm/rnn/while/basic_lstm_cell/MatMul/Enter_grad/b_acc_3ExitAgradients/lstm/rnn/while/basic_lstm_cell/MatMul/Enter_grad/Switch*
T0
�
Qgradients/lstm/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3Wgradients/lstm/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterYgradients/lstm/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1^gradients/Sub*9
_class/
-+loc:@lstm/rnn/while/TensorArrayReadV3/Enter*
source	gradients
�
Wgradients/lstm/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterEnterlstm/rnn/TensorArray_1*
T0*9
_class/
-+loc:@lstm/rnn/while/TensorArrayReadV3/Enter*
is_constant(*
parallel_iterations *6

frame_name(&gradients/lstm/rnn/while/while_context
�
Ygradients/lstm/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1EnterClstm/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
T0*9
_class/
-+loc:@lstm/rnn/while/TensorArrayReadV3/Enter*
is_constant(*
parallel_iterations *6

frame_name(&gradients/lstm/rnn/while/while_context
�
Mgradients/lstm/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/gradient_flowIdentityYgradients/lstm/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1R^gradients/lstm/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*9
_class/
-+loc:@lstm/rnn/while/TensorArrayReadV3/Enter
�
Sgradients/lstm/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3Qgradients/lstm/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3^gradients/lstm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2Mgradients/lstm/rnn/while/basic_lstm_cell/concat_grad/tuple/control_dependencyMgradients/lstm/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/gradient_flow*
T0
j
=gradients/lstm/rnn/while/TensorArrayReadV3/Enter_1_grad/b_accConst*
valueB
 *    *
dtype0
�
?gradients/lstm/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_1Enter=gradients/lstm/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc*
T0*
is_constant( *
parallel_iterations *6

frame_name(&gradients/lstm/rnn/while/while_context
�
?gradients/lstm/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_2Merge?gradients/lstm/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_1Egradients/lstm/rnn/while/TensorArrayReadV3/Enter_1_grad/NextIteration*
T0*
N
�
>gradients/lstm/rnn/while/TensorArrayReadV3/Enter_1_grad/SwitchSwitch?gradients/lstm/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_2gradients/b_count_2*
T0
�
;gradients/lstm/rnn/while/TensorArrayReadV3/Enter_1_grad/AddAdd@gradients/lstm/rnn/while/TensorArrayReadV3/Enter_1_grad/Switch:1Sgradients/lstm/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3*
T0
�
Egradients/lstm/rnn/while/TensorArrayReadV3/Enter_1_grad/NextIterationNextIteration;gradients/lstm/rnn/while/TensorArrayReadV3/Enter_1_grad/Add*
T0
�
?gradients/lstm/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3Exit>gradients/lstm/rnn/while/TensorArrayReadV3/Enter_1_grad/Switch*
T0
�
6gradients/lstm/rnn/while/Switch_4_grad_1/NextIterationNextIterationOgradients/lstm/rnn/while/basic_lstm_cell/concat_grad/tuple/control_dependency_1*
T0
�
tgradients/lstm/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3lstm/rnn/TensorArray_1?gradients/lstm/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3*)
_class
loc:@lstm/rnn/TensorArray_1*
source	gradients
�
pgradients/lstm/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/gradient_flowIdentity?gradients/lstm/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3u^gradients/lstm/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*)
_class
loc:@lstm/rnn/TensorArray_1
�
fgradients/lstm/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3TensorArrayGatherV3tgradients/lstm/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3!lstm/rnn/TensorArrayUnstack/rangepgradients/lstm/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/gradient_flow*
element_shape:*
dtype0
�
cgradients/lstm/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/group_depsNoOpg^gradients/lstm/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3@^gradients/lstm/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3
�
kgradients/lstm/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/control_dependencyIdentityfgradients/lstm/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3d^gradients/lstm/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/group_deps*
T0*y
_classo
mkloc:@gradients/lstm/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3
�
mgradients/lstm/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/control_dependency_1Identity?gradients/lstm/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3d^gradients/lstm/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/group_deps*
T0*R
_classH
FDloc:@gradients/lstm/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3
b
3gradients/lstm/rnn/transpose_grad/InvertPermutationInvertPermutationlstm/rnn/concat*
T0
�
+gradients/lstm/rnn/transpose_grad/transpose	Transposekgradients/lstm/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/control_dependency3gradients/lstm/rnn/transpose_grad/InvertPermutation*
Tperm0*
T0
H
gradients/Reshape_grad/ShapeShapeconcat_2*
T0*
out_type0
�
gradients/Reshape_grad/ReshapeReshape+gradients/lstm/rnn/transpose_grad/transposegradients/Reshape_grad/Shape*
T0*
Tshape0
F
gradients/concat_2_grad/RankConst*
value	B :*
dtype0
]
gradients/concat_2_grad/modFloorModconcat_2/axisgradients/concat_2_grad/Rank*
T0
N
gradients/concat_2_grad/ShapeShapeconcat/concat*
T0*
out_type0
j
gradients/concat_2_grad/ShapeNShapeNconcat/concatconcat_1/concat*
T0*
out_type0*
N
�
$gradients/concat_2_grad/ConcatOffsetConcatOffsetgradients/concat_2_grad/modgradients/concat_2_grad/ShapeN gradients/concat_2_grad/ShapeN:1*
N
�
gradients/concat_2_grad/SliceSlicegradients/Reshape_grad/Reshape$gradients/concat_2_grad/ConcatOffsetgradients/concat_2_grad/ShapeN*
T0*
Index0
�
gradients/concat_2_grad/Slice_1Slicegradients/Reshape_grad/Reshape&gradients/concat_2_grad/ConcatOffset:1 gradients/concat_2_grad/ShapeN:1*
T0*
Index0
r
(gradients/concat_2_grad/tuple/group_depsNoOp^gradients/concat_2_grad/Slice ^gradients/concat_2_grad/Slice_1
�
0gradients/concat_2_grad/tuple/control_dependencyIdentitygradients/concat_2_grad/Slice)^gradients/concat_2_grad/tuple/group_deps*
T0*0
_class&
$"loc:@gradients/concat_2_grad/Slice
�
2gradients/concat_2_grad/tuple/control_dependency_1Identitygradients/concat_2_grad/Slice_1)^gradients/concat_2_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients/concat_2_grad/Slice_1
�
[gradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/Mul_grad/ShapeShapeJmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/BiasAdd*
T0*
out_type0
�
]gradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/Mul_grad/Shape_1ShapeJmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/Sigmoid*
T0*
out_type0
�
kgradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/Mul_grad/BroadcastGradientArgsBroadcastGradientArgs[gradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/Mul_grad/Shape]gradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/Mul_grad/Shape_1*
T0
�
Ygradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/Mul_grad/MulMul0gradients/concat_2_grad/tuple/control_dependencyJmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/Sigmoid*
T0
�
Ygradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/Mul_grad/SumSumYgradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/Mul_grad/Mulkgradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/Mul_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
�
]gradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/Mul_grad/ReshapeReshapeYgradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/Mul_grad/Sum[gradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/Mul_grad/Shape*
T0*
Tshape0
�
[gradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/Mul_grad/Mul_1MulJmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/BiasAdd0gradients/concat_2_grad/tuple/control_dependency*
T0
�
[gradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/Mul_grad/Sum_1Sum[gradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/Mul_grad/Mul_1mgradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/Mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
�
_gradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/Mul_grad/Reshape_1Reshape[gradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/Mul_grad/Sum_1]gradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/Mul_grad/Shape_1*
T0*
Tshape0
�
fgradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/Mul_grad/tuple/group_depsNoOp^^gradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/Mul_grad/Reshape`^gradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/Mul_grad/Reshape_1
�
ngradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/Mul_grad/tuple/control_dependencyIdentity]gradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/Mul_grad/Reshapeg^gradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/Mul_grad/tuple/group_deps*
T0*p
_classf
dbloc:@gradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/Mul_grad/Reshape
�
pgradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/Mul_grad/tuple/control_dependency_1Identity_gradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/Mul_grad/Reshape_1g^gradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/Mul_grad/tuple/group_deps*
T0*r
_classh
fdloc:@gradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/Mul_grad/Reshape_1
�
egradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/Sigmoid_grad/SigmoidGradSigmoidGradJmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/Sigmoidpgradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/Mul_grad/tuple/control_dependency_1*
T0
�
gradients/AddN_7AddNngradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/Mul_grad/tuple/control_dependencyegradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/Sigmoid_grad/SigmoidGrad*
T0*p
_classf
dbloc:@gradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/Mul_grad/Reshape*
N
�
egradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_7*
T0*
data_formatNHWC
�
jgradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_7f^gradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/BiasAdd_grad/BiasAddGrad
�
rgradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_7k^gradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/BiasAdd_grad/tuple/group_deps*
T0*p
_classf
dbloc:@gradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/Mul_grad/Reshape
�
tgradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/BiasAdd_grad/tuple/control_dependency_1Identityegradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/BiasAdd_grad/BiasAddGradk^gradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/BiasAdd_grad/tuple/group_deps*
T0*x
_classn
ljloc:@gradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/BiasAdd_grad/BiasAddGrad
�
_gradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/MatMul_grad/MatMulMatMulrgradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/BiasAdd_grad/tuple/control_dependencyNmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/kernel/read*
transpose_b(*
T0*
transpose_a( 
�
agradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/MatMul_grad/MatMul_1MatMulFmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/Mulrgradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(
�
igradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/MatMul_grad/tuple/group_depsNoOp`^gradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/MatMul_grad/MatMulb^gradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/MatMul_grad/MatMul_1
�
qgradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/MatMul_grad/tuple/control_dependencyIdentity_gradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/MatMul_grad/MatMulj^gradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/MatMul_grad/tuple/group_deps*
T0*r
_classh
fdloc:@gradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/MatMul_grad/MatMul
�
sgradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/MatMul_grad/tuple/control_dependency_1Identityagradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/MatMul_grad/MatMul_1j^gradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/MatMul_grad/tuple/group_deps*
T0*t
_classj
hfloc:@gradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/MatMul_grad/MatMul_1
�
[gradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/Mul_grad/ShapeShapeJmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/BiasAdd*
T0*
out_type0
�
]gradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/Mul_grad/Shape_1ShapeJmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/Sigmoid*
T0*
out_type0
�
kgradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/Mul_grad/BroadcastGradientArgsBroadcastGradientArgs[gradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/Mul_grad/Shape]gradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/Mul_grad/Shape_1*
T0
�
Ygradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/Mul_grad/MulMulqgradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/MatMul_grad/tuple/control_dependencyJmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/Sigmoid*
T0
�
Ygradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/Mul_grad/SumSumYgradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/Mul_grad/Mulkgradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/Mul_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
�
]gradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/Mul_grad/ReshapeReshapeYgradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/Mul_grad/Sum[gradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/Mul_grad/Shape*
T0*
Tshape0
�
[gradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/Mul_grad/Mul_1MulJmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/BiasAddqgradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/MatMul_grad/tuple/control_dependency*
T0
�
[gradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/Mul_grad/Sum_1Sum[gradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/Mul_grad/Mul_1mgradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/Mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
�
_gradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/Mul_grad/Reshape_1Reshape[gradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/Mul_grad/Sum_1]gradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/Mul_grad/Shape_1*
T0*
Tshape0
�
fgradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/Mul_grad/tuple/group_depsNoOp^^gradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/Mul_grad/Reshape`^gradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/Mul_grad/Reshape_1
�
ngradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/Mul_grad/tuple/control_dependencyIdentity]gradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/Mul_grad/Reshapeg^gradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/Mul_grad/tuple/group_deps*
T0*p
_classf
dbloc:@gradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/Mul_grad/Reshape
�
pgradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/Mul_grad/tuple/control_dependency_1Identity_gradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/Mul_grad/Reshape_1g^gradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/Mul_grad/tuple/group_deps*
T0*r
_classh
fdloc:@gradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/Mul_grad/Reshape_1
�
egradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/Sigmoid_grad/SigmoidGradSigmoidGradJmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/Sigmoidpgradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/Mul_grad/tuple/control_dependency_1*
T0
�
gradients/AddN_8AddNngradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/Mul_grad/tuple/control_dependencyegradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/Sigmoid_grad/SigmoidGrad*
T0*p
_classf
dbloc:@gradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/Mul_grad/Reshape*
N
�
egradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/BiasAdd_grad/BiasAddGradBiasAddGradgradients/AddN_8*
T0*
data_formatNHWC
�
jgradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/BiasAdd_grad/tuple/group_depsNoOp^gradients/AddN_8f^gradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/BiasAdd_grad/BiasAddGrad
�
rgradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/BiasAdd_grad/tuple/control_dependencyIdentitygradients/AddN_8k^gradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/BiasAdd_grad/tuple/group_deps*
T0*p
_classf
dbloc:@gradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/Mul_grad/Reshape
�
tgradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/BiasAdd_grad/tuple/control_dependency_1Identityegradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/BiasAdd_grad/BiasAddGradk^gradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/BiasAdd_grad/tuple/group_deps*
T0*x
_classn
ljloc:@gradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/BiasAdd_grad/BiasAddGrad
�
_gradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/MatMul_grad/MatMulMatMulrgradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/BiasAdd_grad/tuple/control_dependencyNmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/kernel/read*
transpose_b(*
T0*
transpose_a( 
�
agradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/MatMul_grad/MatMul_1MatMul-main_graph_0_encoder0/Flatten/flatten/Reshapergradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(
�
igradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/MatMul_grad/tuple/group_depsNoOp`^gradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/MatMul_grad/MatMulb^gradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/MatMul_grad/MatMul_1
�
qgradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/MatMul_grad/tuple/control_dependencyIdentity_gradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/MatMul_grad/MatMulj^gradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/MatMul_grad/tuple/group_deps*
T0*r
_classh
fdloc:@gradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/MatMul_grad/MatMul
�
sgradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/MatMul_grad/tuple/control_dependency_1Identityagradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/MatMul_grad/MatMul_1j^gradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/MatMul_grad/tuple/group_deps*
T0*t
_classj
hfloc:@gradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/MatMul_grad/MatMul_1
�
Bgradients/main_graph_0_encoder0/Flatten/flatten/Reshape_grad/ShapeShape main_graph_0_encoder0/conv_1/Elu*
T0*
out_type0
�
Dgradients/main_graph_0_encoder0/Flatten/flatten/Reshape_grad/ReshapeReshapeqgradients/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/MatMul_grad/tuple/control_dependencyBgradients/main_graph_0_encoder0/Flatten/flatten/Reshape_grad/Shape*
T0*
Tshape0
�
7gradients/main_graph_0_encoder0/conv_1/Elu_grad/EluGradEluGradDgradients/main_graph_0_encoder0/Flatten/flatten/Reshape_grad/Reshape main_graph_0_encoder0/conv_1/Elu*
T0
�
?gradients/main_graph_0_encoder0/conv_1/BiasAdd_grad/BiasAddGradBiasAddGrad7gradients/main_graph_0_encoder0/conv_1/Elu_grad/EluGrad*
T0*
data_formatNHWC
�
Dgradients/main_graph_0_encoder0/conv_1/BiasAdd_grad/tuple/group_depsNoOp@^gradients/main_graph_0_encoder0/conv_1/BiasAdd_grad/BiasAddGrad8^gradients/main_graph_0_encoder0/conv_1/Elu_grad/EluGrad
�
Lgradients/main_graph_0_encoder0/conv_1/BiasAdd_grad/tuple/control_dependencyIdentity7gradients/main_graph_0_encoder0/conv_1/Elu_grad/EluGradE^gradients/main_graph_0_encoder0/conv_1/BiasAdd_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/main_graph_0_encoder0/conv_1/Elu_grad/EluGrad
�
Ngradients/main_graph_0_encoder0/conv_1/BiasAdd_grad/tuple/control_dependency_1Identity?gradients/main_graph_0_encoder0/conv_1/BiasAdd_grad/BiasAddGradE^gradients/main_graph_0_encoder0/conv_1/BiasAdd_grad/tuple/group_deps*
T0*R
_classH
FDloc:@gradients/main_graph_0_encoder0/conv_1/BiasAdd_grad/BiasAddGrad
�
9gradients/main_graph_0_encoder0/conv_1/Conv2D_grad/ShapeNShapeN main_graph_0_encoder0/conv_0/Elu(main_graph_0_encoder0/conv_1/kernel/read*
T0*
out_type0*
N
�
Fgradients/main_graph_0_encoder0/conv_1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput9gradients/main_graph_0_encoder0/conv_1/Conv2D_grad/ShapeN(main_graph_0_encoder0/conv_1/kernel/readLgradients/main_graph_0_encoder0/conv_1/BiasAdd_grad/tuple/control_dependency*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingVALID
�
Ggradients/main_graph_0_encoder0/conv_1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilter main_graph_0_encoder0/conv_0/Elu;gradients/main_graph_0_encoder0/conv_1/Conv2D_grad/ShapeN:1Lgradients/main_graph_0_encoder0/conv_1/BiasAdd_grad/tuple/control_dependency*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingVALID
�
Cgradients/main_graph_0_encoder0/conv_1/Conv2D_grad/tuple/group_depsNoOpH^gradients/main_graph_0_encoder0/conv_1/Conv2D_grad/Conv2DBackpropFilterG^gradients/main_graph_0_encoder0/conv_1/Conv2D_grad/Conv2DBackpropInput
�
Kgradients/main_graph_0_encoder0/conv_1/Conv2D_grad/tuple/control_dependencyIdentityFgradients/main_graph_0_encoder0/conv_1/Conv2D_grad/Conv2DBackpropInputD^gradients/main_graph_0_encoder0/conv_1/Conv2D_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients/main_graph_0_encoder0/conv_1/Conv2D_grad/Conv2DBackpropInput
�
Mgradients/main_graph_0_encoder0/conv_1/Conv2D_grad/tuple/control_dependency_1IdentityGgradients/main_graph_0_encoder0/conv_1/Conv2D_grad/Conv2DBackpropFilterD^gradients/main_graph_0_encoder0/conv_1/Conv2D_grad/tuple/group_deps*
T0*Z
_classP
NLloc:@gradients/main_graph_0_encoder0/conv_1/Conv2D_grad/Conv2DBackpropFilter
�
7gradients/main_graph_0_encoder0/conv_0/Elu_grad/EluGradEluGradKgradients/main_graph_0_encoder0/conv_1/Conv2D_grad/tuple/control_dependency main_graph_0_encoder0/conv_0/Elu*
T0
�
?gradients/main_graph_0_encoder0/conv_0/BiasAdd_grad/BiasAddGradBiasAddGrad7gradients/main_graph_0_encoder0/conv_0/Elu_grad/EluGrad*
T0*
data_formatNHWC
�
Dgradients/main_graph_0_encoder0/conv_0/BiasAdd_grad/tuple/group_depsNoOp@^gradients/main_graph_0_encoder0/conv_0/BiasAdd_grad/BiasAddGrad8^gradients/main_graph_0_encoder0/conv_0/Elu_grad/EluGrad
�
Lgradients/main_graph_0_encoder0/conv_0/BiasAdd_grad/tuple/control_dependencyIdentity7gradients/main_graph_0_encoder0/conv_0/Elu_grad/EluGradE^gradients/main_graph_0_encoder0/conv_0/BiasAdd_grad/tuple/group_deps*
T0*J
_class@
><loc:@gradients/main_graph_0_encoder0/conv_0/Elu_grad/EluGrad
�
Ngradients/main_graph_0_encoder0/conv_0/BiasAdd_grad/tuple/control_dependency_1Identity?gradients/main_graph_0_encoder0/conv_0/BiasAdd_grad/BiasAddGradE^gradients/main_graph_0_encoder0/conv_0/BiasAdd_grad/tuple/group_deps*
T0*R
_classH
FDloc:@gradients/main_graph_0_encoder0/conv_0/BiasAdd_grad/BiasAddGrad
�
9gradients/main_graph_0_encoder0/conv_0/Conv2D_grad/ShapeNShapeNvisual_observation_0(main_graph_0_encoder0/conv_0/kernel/read*
T0*
out_type0*
N
�
Fgradients/main_graph_0_encoder0/conv_0/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput9gradients/main_graph_0_encoder0/conv_0/Conv2D_grad/ShapeN(main_graph_0_encoder0/conv_0/kernel/readLgradients/main_graph_0_encoder0/conv_0/BiasAdd_grad/tuple/control_dependency*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingVALID
�
Ggradients/main_graph_0_encoder0/conv_0/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltervisual_observation_0;gradients/main_graph_0_encoder0/conv_0/Conv2D_grad/ShapeN:1Lgradients/main_graph_0_encoder0/conv_0/BiasAdd_grad/tuple/control_dependency*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingVALID
�
Cgradients/main_graph_0_encoder0/conv_0/Conv2D_grad/tuple/group_depsNoOpH^gradients/main_graph_0_encoder0/conv_0/Conv2D_grad/Conv2DBackpropFilterG^gradients/main_graph_0_encoder0/conv_0/Conv2D_grad/Conv2DBackpropInput
�
Kgradients/main_graph_0_encoder0/conv_0/Conv2D_grad/tuple/control_dependencyIdentityFgradients/main_graph_0_encoder0/conv_0/Conv2D_grad/Conv2DBackpropInputD^gradients/main_graph_0_encoder0/conv_0/Conv2D_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients/main_graph_0_encoder0/conv_0/Conv2D_grad/Conv2DBackpropInput
�
Mgradients/main_graph_0_encoder0/conv_0/Conv2D_grad/tuple/control_dependency_1IdentityGgradients/main_graph_0_encoder0/conv_0/Conv2D_grad/Conv2DBackpropFilterD^gradients/main_graph_0_encoder0/conv_0/Conv2D_grad/tuple/group_deps*
T0*Z
_classP
NLloc:@gradients/main_graph_0_encoder0/conv_0/Conv2D_grad/Conv2DBackpropFilter
:
gradients_1/ShapeConst*
valueB *
dtype0
B
gradients_1/grad_ys_0Const*
valueB
 *  �?*
dtype0
]
gradients_1/FillFillgradients_1/Shapegradients_1/grad_ys_0*
T0*

index_type0
=
gradients_1/f_countConst*
value	B : *
dtype0
�
gradients_1/f_count_1Entergradients_1/f_count*
T0*
is_constant( *
parallel_iterations *,

frame_namelstm/rnn/while/while_context
^
gradients_1/MergeMergegradients_1/f_count_1gradients_1/NextIteration*
T0*
N
Q
gradients_1/SwitchSwitchgradients_1/Mergelstm/rnn/while/LoopCond*
T0
U
gradients_1/Add/yConst^lstm/rnn/while/Identity*
value	B :*
dtype0
H
gradients_1/AddAddgradients_1/Switch:1gradients_1/Add/y*
T0
�
gradients_1/NextIterationNextIterationgradients_1/Addb^gradients_1/lstm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2X^gradients_1/lstm/rnn/while/basic_lstm_cell/Add_1_grad/BroadcastGradientArgs/StackPushV2Z^gradients_1/lstm/rnn/while/basic_lstm_cell/Add_1_grad/BroadcastGradientArgs/StackPushV2_1V^gradients_1/lstm/rnn/while/basic_lstm_cell/Add_grad/BroadcastGradientArgs/StackPushV2X^gradients_1/lstm/rnn/while/basic_lstm_cell/Add_grad/BroadcastGradientArgs/StackPushV2_1L^gradients_1/lstm/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/StackPushV2X^gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/BroadcastGradientArgs/StackPushV2Z^gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/BroadcastGradientArgs/StackPushV2_1F^gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/StackPushV2H^gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/StackPushV2X^gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/BroadcastGradientArgs/StackPushV2Z^gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/BroadcastGradientArgs/StackPushV2_1F^gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/StackPushV2H^gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/StackPushV2V^gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_grad/BroadcastGradientArgs/StackPushV2X^gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_grad/BroadcastGradientArgs/StackPushV2_1D^gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_grad/Mul/StackPushV2F^gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/StackPushV2J^gradients_1/lstm/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPushV2L^gradients_1/lstm/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPushV2_1*
T0
:
gradients_1/f_count_2Exitgradients_1/Switch*
T0
=
gradients_1/b_countConst*
value	B :*
dtype0
�
gradients_1/b_count_1Entergradients_1/f_count_2*
T0*
is_constant( *
parallel_iterations *8

frame_name*(gradients_1/lstm/rnn/while/while_context
b
gradients_1/Merge_1Mergegradients_1/b_count_1gradients_1/NextIteration_1*
T0*
N
f
gradients_1/GreaterEqualGreaterEqualgradients_1/Merge_1gradients_1/GreaterEqual/Enter*
T0
�
gradients_1/GreaterEqual/EnterEntergradients_1/b_count*
T0*
is_constant(*
parallel_iterations *8

frame_name*(gradients_1/lstm/rnn/while/while_context
;
gradients_1/b_count_2LoopCondgradients_1/GreaterEqual
S
gradients_1/Switch_1Switchgradients_1/Merge_1gradients_1/b_count_2*
T0
W
gradients_1/SubSubgradients_1/Switch_1:1gradients_1/GreaterEqual/Enter*
T0
�
gradients_1/NextIteration_1NextIterationgradients_1/Sub]^gradients_1/lstm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/b_sync*
T0
<
gradients_1/b_count_3Exitgradients_1/Switch_1*
T0
<
gradients_1/sub_3_grad/NegNeggradients_1/Fill*
T0
_
'gradients_1/sub_3_grad/tuple/group_depsNoOp^gradients_1/Fill^gradients_1/sub_3_grad/Neg
�
/gradients_1/sub_3_grad/tuple/control_dependencyIdentitygradients_1/Fill(^gradients_1/sub_3_grad/tuple/group_deps*
T0*#
_class
loc:@gradients_1/Fill
�
1gradients_1/sub_3_grad/tuple/control_dependency_1Identitygradients_1/sub_3_grad/Neg(^gradients_1/sub_3_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients_1/sub_3_grad/Neg
a
'gradients_1/add_9_grad/tuple/group_depsNoOp0^gradients_1/sub_3_grad/tuple/control_dependency
�
/gradients_1/add_9_grad/tuple/control_dependencyIdentity/gradients_1/sub_3_grad/tuple/control_dependency(^gradients_1/add_9_grad/tuple/group_deps*
T0*#
_class
loc:@gradients_1/Fill
�
1gradients_1/add_9_grad/tuple/control_dependency_1Identity/gradients_1/sub_3_grad/tuple/control_dependency(^gradients_1/add_9_grad/tuple/group_deps*
T0*#
_class
loc:@gradients_1/Fill
e
gradients_1/mul_5_grad/MulMul1gradients_1/sub_3_grad/tuple/control_dependency_1Mean_4*
T0
r
gradients_1/mul_5_grad/Mul_1Mul1gradients_1/sub_3_grad/tuple/control_dependency_1PolynomialDecay_2*
T0
k
'gradients_1/mul_5_grad/tuple/group_depsNoOp^gradients_1/mul_5_grad/Mul^gradients_1/mul_5_grad/Mul_1
�
/gradients_1/mul_5_grad/tuple/control_dependencyIdentitygradients_1/mul_5_grad/Mul(^gradients_1/mul_5_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients_1/mul_5_grad/Mul
�
1gradients_1/mul_5_grad/tuple/control_dependency_1Identitygradients_1/mul_5_grad/Mul_1(^gradients_1/mul_5_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients_1/mul_5_grad/Mul_1
[
gradients_1/Neg_3_grad/NegNeg/gradients_1/add_9_grad/tuple/control_dependency*
T0
e
gradients_1/mul_4_grad/MulMul1gradients_1/add_9_grad/tuple/control_dependency_1Mean_2*
T0
h
gradients_1/mul_4_grad/Mul_1Mul1gradients_1/add_9_grad/tuple/control_dependency_1mul_4/x*
T0
k
'gradients_1/mul_4_grad/tuple/group_depsNoOp^gradients_1/mul_4_grad/Mul^gradients_1/mul_4_grad/Mul_1
�
/gradients_1/mul_4_grad/tuple/control_dependencyIdentitygradients_1/mul_4_grad/Mul(^gradients_1/mul_4_grad/tuple/group_deps*
T0*-
_class#
!loc:@gradients_1/mul_4_grad/Mul
�
1gradients_1/mul_4_grad/tuple/control_dependency_1Identitygradients_1/mul_4_grad/Mul_1(^gradients_1/mul_4_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients_1/mul_4_grad/Mul_1
S
%gradients_1/Mean_4_grad/Reshape/shapeConst*
valueB:*
dtype0
�
gradients_1/Mean_4_grad/ReshapeReshape1gradients_1/mul_5_grad/tuple/control_dependency_1%gradients_1/Mean_4_grad/Reshape/shape*
T0*
Tshape0
U
gradients_1/Mean_4_grad/ShapeShapeDynamicPartition_2:1*
T0*
out_type0

gradients_1/Mean_4_grad/TileTilegradients_1/Mean_4_grad/Reshapegradients_1/Mean_4_grad/Shape*

Tmultiples0*
T0
W
gradients_1/Mean_4_grad/Shape_1ShapeDynamicPartition_2:1*
T0*
out_type0
H
gradients_1/Mean_4_grad/Shape_2Const*
valueB *
dtype0
K
gradients_1/Mean_4_grad/ConstConst*
valueB: *
dtype0
�
gradients_1/Mean_4_grad/ProdProdgradients_1/Mean_4_grad/Shape_1gradients_1/Mean_4_grad/Const*

Tidx0*
	keep_dims( *
T0
M
gradients_1/Mean_4_grad/Const_1Const*
valueB: *
dtype0
�
gradients_1/Mean_4_grad/Prod_1Prodgradients_1/Mean_4_grad/Shape_2gradients_1/Mean_4_grad/Const_1*

Tidx0*
	keep_dims( *
T0
K
!gradients_1/Mean_4_grad/Maximum/yConst*
value	B :*
dtype0
v
gradients_1/Mean_4_grad/MaximumMaximumgradients_1/Mean_4_grad/Prod_1!gradients_1/Mean_4_grad/Maximum/y*
T0
t
 gradients_1/Mean_4_grad/floordivFloorDivgradients_1/Mean_4_grad/Prodgradients_1/Mean_4_grad/Maximum*
T0
n
gradients_1/Mean_4_grad/CastCast gradients_1/Mean_4_grad/floordiv*

SrcT0*
Truncate( *

DstT0
o
gradients_1/Mean_4_grad/truedivRealDivgradients_1/Mean_4_grad/Tilegradients_1/Mean_4_grad/Cast*
T0
Z
%gradients_1/Mean_3_grad/Reshape/shapeConst*
valueB"      *
dtype0
�
gradients_1/Mean_3_grad/ReshapeReshapegradients_1/Neg_3_grad/Neg%gradients_1/Mean_3_grad/Reshape/shape*
T0*
Tshape0
U
gradients_1/Mean_3_grad/ShapeShapeDynamicPartition_1:1*
T0*
out_type0

gradients_1/Mean_3_grad/TileTilegradients_1/Mean_3_grad/Reshapegradients_1/Mean_3_grad/Shape*

Tmultiples0*
T0
W
gradients_1/Mean_3_grad/Shape_1ShapeDynamicPartition_1:1*
T0*
out_type0
H
gradients_1/Mean_3_grad/Shape_2Const*
valueB *
dtype0
K
gradients_1/Mean_3_grad/ConstConst*
valueB: *
dtype0
�
gradients_1/Mean_3_grad/ProdProdgradients_1/Mean_3_grad/Shape_1gradients_1/Mean_3_grad/Const*

Tidx0*
	keep_dims( *
T0
M
gradients_1/Mean_3_grad/Const_1Const*
valueB: *
dtype0
�
gradients_1/Mean_3_grad/Prod_1Prodgradients_1/Mean_3_grad/Shape_2gradients_1/Mean_3_grad/Const_1*

Tidx0*
	keep_dims( *
T0
K
!gradients_1/Mean_3_grad/Maximum/yConst*
value	B :*
dtype0
v
gradients_1/Mean_3_grad/MaximumMaximumgradients_1/Mean_3_grad/Prod_1!gradients_1/Mean_3_grad/Maximum/y*
T0
t
 gradients_1/Mean_3_grad/floordivFloorDivgradients_1/Mean_3_grad/Prodgradients_1/Mean_3_grad/Maximum*
T0
n
gradients_1/Mean_3_grad/CastCast gradients_1/Mean_3_grad/floordiv*

SrcT0*
Truncate( *

DstT0
o
gradients_1/Mean_3_grad/truedivRealDivgradients_1/Mean_3_grad/Tilegradients_1/Mean_3_grad/Cast*
T0
S
%gradients_1/Mean_2_grad/Reshape/shapeConst*
valueB:*
dtype0
�
gradients_1/Mean_2_grad/ReshapeReshape1gradients_1/mul_4_grad/tuple/control_dependency_1%gradients_1/Mean_2_grad/Reshape/shape*
T0*
Tshape0
K
gradients_1/Mean_2_grad/ConstConst*
valueB:*
dtype0

gradients_1/Mean_2_grad/TileTilegradients_1/Mean_2_grad/Reshapegradients_1/Mean_2_grad/Const*

Tmultiples0*
T0
L
gradients_1/Mean_2_grad/Const_1Const*
valueB
 *  �?*
dtype0
r
gradients_1/Mean_2_grad/truedivRealDivgradients_1/Mean_2_grad/Tilegradients_1/Mean_2_grad/Const_1*
T0
@
gradients_1/zeros_like	ZerosLikeDynamicPartition_2*
T0
Q
)gradients_1/DynamicPartition_2_grad/ShapeShapeCast*
T0*
out_type0
W
)gradients_1/DynamicPartition_2_grad/ConstConst*
valueB: *
dtype0
�
(gradients_1/DynamicPartition_2_grad/ProdProd)gradients_1/DynamicPartition_2_grad/Shape)gradients_1/DynamicPartition_2_grad/Const*

Tidx0*
	keep_dims( *
T0
Y
/gradients_1/DynamicPartition_2_grad/range/startConst*
value	B : *
dtype0
Y
/gradients_1/DynamicPartition_2_grad/range/deltaConst*
value	B :*
dtype0
�
)gradients_1/DynamicPartition_2_grad/rangeRange/gradients_1/DynamicPartition_2_grad/range/start(gradients_1/DynamicPartition_2_grad/Prod/gradients_1/DynamicPartition_2_grad/range/delta*

Tidx0
�
+gradients_1/DynamicPartition_2_grad/ReshapeReshape)gradients_1/DynamicPartition_2_grad/range)gradients_1/DynamicPartition_2_grad/Shape*
T0*
Tshape0
�
4gradients_1/DynamicPartition_2_grad/DynamicPartitionDynamicPartition+gradients_1/DynamicPartition_2_grad/ReshapeCast*
num_partitions*
T0
�
1gradients_1/DynamicPartition_2_grad/DynamicStitchDynamicStitch4gradients_1/DynamicPartition_2_grad/DynamicPartition6gradients_1/DynamicPartition_2_grad/DynamicPartition:1gradients_1/zeros_likegradients_1/Mean_4_grad/truediv*
T0*
N
T
+gradients_1/DynamicPartition_2_grad/Shape_1ShapeSum_2*
T0*
out_type0
�
-gradients_1/DynamicPartition_2_grad/Reshape_1Reshape1gradients_1/DynamicPartition_2_grad/DynamicStitch+gradients_1/DynamicPartition_2_grad/Shape_1*
T0*
Tshape0
B
gradients_1/zeros_like_1	ZerosLikeDynamicPartition_1*
T0
Q
)gradients_1/DynamicPartition_1_grad/ShapeShapeCast*
T0*
out_type0
W
)gradients_1/DynamicPartition_1_grad/ConstConst*
valueB: *
dtype0
�
(gradients_1/DynamicPartition_1_grad/ProdProd)gradients_1/DynamicPartition_1_grad/Shape)gradients_1/DynamicPartition_1_grad/Const*

Tidx0*
	keep_dims( *
T0
Y
/gradients_1/DynamicPartition_1_grad/range/startConst*
value	B : *
dtype0
Y
/gradients_1/DynamicPartition_1_grad/range/deltaConst*
value	B :*
dtype0
�
)gradients_1/DynamicPartition_1_grad/rangeRange/gradients_1/DynamicPartition_1_grad/range/start(gradients_1/DynamicPartition_1_grad/Prod/gradients_1/DynamicPartition_1_grad/range/delta*

Tidx0
�
+gradients_1/DynamicPartition_1_grad/ReshapeReshape)gradients_1/DynamicPartition_1_grad/range)gradients_1/DynamicPartition_1_grad/Shape*
T0*
Tshape0
�
4gradients_1/DynamicPartition_1_grad/DynamicPartitionDynamicPartition+gradients_1/DynamicPartition_1_grad/ReshapeCast*
num_partitions*
T0
�
1gradients_1/DynamicPartition_1_grad/DynamicStitchDynamicStitch4gradients_1/DynamicPartition_1_grad/DynamicPartition6gradients_1/DynamicPartition_1_grad/DynamicPartition:1gradients_1/zeros_like_1gradients_1/Mean_3_grad/truediv*
T0*
N
V
+gradients_1/DynamicPartition_1_grad/Shape_1ShapeMinimum*
T0*
out_type0
�
-gradients_1/DynamicPartition_1_grad/Reshape_1Reshape1gradients_1/DynamicPartition_1_grad/DynamicStitch+gradients_1/DynamicPartition_1_grad/Shape_1*
T0*
Tshape0
p
%gradients_1/Mean_2/input_grad/unstackUnpackgradients_1/Mean_2_grad/truediv*
T0*	
num*

axis 
E
gradients_1/Sum_2_grad/ShapeShapestack*
T0*
out_type0
v
gradients_1/Sum_2_grad/SizeConst*/
_class%
#!loc:@gradients_1/Sum_2_grad/Shape*
value	B :*
dtype0
�
gradients_1/Sum_2_grad/addAddV2Sum_2/reduction_indicesgradients_1/Sum_2_grad/Size*
T0*/
_class%
#!loc:@gradients_1/Sum_2_grad/Shape
�
gradients_1/Sum_2_grad/modFloorModgradients_1/Sum_2_grad/addgradients_1/Sum_2_grad/Size*
T0*/
_class%
#!loc:@gradients_1/Sum_2_grad/Shape
x
gradients_1/Sum_2_grad/Shape_1Const*/
_class%
#!loc:@gradients_1/Sum_2_grad/Shape*
valueB *
dtype0
}
"gradients_1/Sum_2_grad/range/startConst*/
_class%
#!loc:@gradients_1/Sum_2_grad/Shape*
value	B : *
dtype0
}
"gradients_1/Sum_2_grad/range/deltaConst*/
_class%
#!loc:@gradients_1/Sum_2_grad/Shape*
value	B :*
dtype0
�
gradients_1/Sum_2_grad/rangeRange"gradients_1/Sum_2_grad/range/startgradients_1/Sum_2_grad/Size"gradients_1/Sum_2_grad/range/delta*

Tidx0*/
_class%
#!loc:@gradients_1/Sum_2_grad/Shape
|
!gradients_1/Sum_2_grad/Fill/valueConst*/
_class%
#!loc:@gradients_1/Sum_2_grad/Shape*
value	B :*
dtype0
�
gradients_1/Sum_2_grad/FillFillgradients_1/Sum_2_grad/Shape_1!gradients_1/Sum_2_grad/Fill/value*
T0*/
_class%
#!loc:@gradients_1/Sum_2_grad/Shape*

index_type0
�
$gradients_1/Sum_2_grad/DynamicStitchDynamicStitchgradients_1/Sum_2_grad/rangegradients_1/Sum_2_grad/modgradients_1/Sum_2_grad/Shapegradients_1/Sum_2_grad/Fill*
T0*/
_class%
#!loc:@gradients_1/Sum_2_grad/Shape*
N
{
 gradients_1/Sum_2_grad/Maximum/yConst*/
_class%
#!loc:@gradients_1/Sum_2_grad/Shape*
value	B :*
dtype0
�
gradients_1/Sum_2_grad/MaximumMaximum$gradients_1/Sum_2_grad/DynamicStitch gradients_1/Sum_2_grad/Maximum/y*
T0*/
_class%
#!loc:@gradients_1/Sum_2_grad/Shape
�
gradients_1/Sum_2_grad/floordivFloorDivgradients_1/Sum_2_grad/Shapegradients_1/Sum_2_grad/Maximum*
T0*/
_class%
#!loc:@gradients_1/Sum_2_grad/Shape
�
gradients_1/Sum_2_grad/ReshapeReshape-gradients_1/DynamicPartition_2_grad/Reshape_1$gradients_1/Sum_2_grad/DynamicStitch*
T0*
Tshape0

gradients_1/Sum_2_grad/TileTilegradients_1/Sum_2_grad/Reshapegradients_1/Sum_2_grad/floordiv*

Tmultiples0*
T0
G
gradients_1/Minimum_grad/ShapeShapemul_2*
T0*
out_type0
I
 gradients_1/Minimum_grad/Shape_1Shapemul_3*
T0*
out_type0
q
 gradients_1/Minimum_grad/Shape_2Shape-gradients_1/DynamicPartition_1_grad/Reshape_1*
T0*
out_type0
Q
$gradients_1/Minimum_grad/zeros/ConstConst*
valueB
 *    *
dtype0
�
gradients_1/Minimum_grad/zerosFill gradients_1/Minimum_grad/Shape_2$gradients_1/Minimum_grad/zeros/Const*
T0*

index_type0
F
"gradients_1/Minimum_grad/LessEqual	LessEqualmul_2mul_3*
T0
�
.gradients_1/Minimum_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/Minimum_grad/Shape gradients_1/Minimum_grad/Shape_1*
T0
�
gradients_1/Minimum_grad/SelectSelect"gradients_1/Minimum_grad/LessEqual-gradients_1/DynamicPartition_1_grad/Reshape_1gradients_1/Minimum_grad/zeros*
T0
�
gradients_1/Minimum_grad/SumSumgradients_1/Minimum_grad/Select.gradients_1/Minimum_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
�
 gradients_1/Minimum_grad/ReshapeReshapegradients_1/Minimum_grad/Sumgradients_1/Minimum_grad/Shape*
T0*
Tshape0
�
!gradients_1/Minimum_grad/Select_1Select"gradients_1/Minimum_grad/LessEqualgradients_1/Minimum_grad/zeros-gradients_1/DynamicPartition_1_grad/Reshape_1*
T0
�
gradients_1/Minimum_grad/Sum_1Sum!gradients_1/Minimum_grad/Select_10gradients_1/Minimum_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
�
"gradients_1/Minimum_grad/Reshape_1Reshapegradients_1/Minimum_grad/Sum_1 gradients_1/Minimum_grad/Shape_1*
T0*
Tshape0
y
)gradients_1/Minimum_grad/tuple/group_depsNoOp!^gradients_1/Minimum_grad/Reshape#^gradients_1/Minimum_grad/Reshape_1
�
1gradients_1/Minimum_grad/tuple/control_dependencyIdentity gradients_1/Minimum_grad/Reshape*^gradients_1/Minimum_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients_1/Minimum_grad/Reshape
�
3gradients_1/Minimum_grad/tuple/control_dependency_1Identity"gradients_1/Minimum_grad/Reshape_1*^gradients_1/Minimum_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients_1/Minimum_grad/Reshape_1
S
%gradients_1/Mean_1_grad/Reshape/shapeConst*
valueB:*
dtype0
�
gradients_1/Mean_1_grad/ReshapeReshape%gradients_1/Mean_2/input_grad/unstack%gradients_1/Mean_1_grad/Reshape/shape*
T0*
Tshape0
S
gradients_1/Mean_1_grad/ShapeShapeDynamicPartition:1*
T0*
out_type0

gradients_1/Mean_1_grad/TileTilegradients_1/Mean_1_grad/Reshapegradients_1/Mean_1_grad/Shape*

Tmultiples0*
T0
U
gradients_1/Mean_1_grad/Shape_1ShapeDynamicPartition:1*
T0*
out_type0
H
gradients_1/Mean_1_grad/Shape_2Const*
valueB *
dtype0
K
gradients_1/Mean_1_grad/ConstConst*
valueB: *
dtype0
�
gradients_1/Mean_1_grad/ProdProdgradients_1/Mean_1_grad/Shape_1gradients_1/Mean_1_grad/Const*

Tidx0*
	keep_dims( *
T0
M
gradients_1/Mean_1_grad/Const_1Const*
valueB: *
dtype0
�
gradients_1/Mean_1_grad/Prod_1Prodgradients_1/Mean_1_grad/Shape_2gradients_1/Mean_1_grad/Const_1*

Tidx0*
	keep_dims( *
T0
K
!gradients_1/Mean_1_grad/Maximum/yConst*
value	B :*
dtype0
v
gradients_1/Mean_1_grad/MaximumMaximumgradients_1/Mean_1_grad/Prod_1!gradients_1/Mean_1_grad/Maximum/y*
T0
t
 gradients_1/Mean_1_grad/floordivFloorDivgradients_1/Mean_1_grad/Prodgradients_1/Mean_1_grad/Maximum*
T0
n
gradients_1/Mean_1_grad/CastCast gradients_1/Mean_1_grad/floordiv*

SrcT0*
Truncate( *

DstT0
o
gradients_1/Mean_1_grad/truedivRealDivgradients_1/Mean_1_grad/Tilegradients_1/Mean_1_grad/Cast*
T0
e
gradients_1/stack_grad/unstackUnpackgradients_1/Sum_2_grad/Tile*
T0*	
num*

axis
C
gradients_1/mul_2_grad/ShapeShapeExp*
T0*
out_type0
L
gradients_1/mul_2_grad/Shape_1Shape
ExpandDims*
T0*
out_type0
�
,gradients_1/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/mul_2_grad/Shapegradients_1/mul_2_grad/Shape_1*
T0
i
gradients_1/mul_2_grad/MulMul1gradients_1/Minimum_grad/tuple/control_dependency
ExpandDims*
T0
�
gradients_1/mul_2_grad/SumSumgradients_1/mul_2_grad/Mul,gradients_1/mul_2_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
z
gradients_1/mul_2_grad/ReshapeReshapegradients_1/mul_2_grad/Sumgradients_1/mul_2_grad/Shape*
T0*
Tshape0
d
gradients_1/mul_2_grad/Mul_1MulExp1gradients_1/Minimum_grad/tuple/control_dependency*
T0
�
gradients_1/mul_2_grad/Sum_1Sumgradients_1/mul_2_grad/Mul_1.gradients_1/mul_2_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
�
 gradients_1/mul_2_grad/Reshape_1Reshapegradients_1/mul_2_grad/Sum_1gradients_1/mul_2_grad/Shape_1*
T0*
Tshape0
s
'gradients_1/mul_2_grad/tuple/group_depsNoOp^gradients_1/mul_2_grad/Reshape!^gradients_1/mul_2_grad/Reshape_1
�
/gradients_1/mul_2_grad/tuple/control_dependencyIdentitygradients_1/mul_2_grad/Reshape(^gradients_1/mul_2_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients_1/mul_2_grad/Reshape
�
1gradients_1/mul_2_grad/tuple/control_dependency_1Identity gradients_1/mul_2_grad/Reshape_1(^gradients_1/mul_2_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients_1/mul_2_grad/Reshape_1
O
gradients_1/mul_3_grad/ShapeShapeclip_by_value_1*
T0*
out_type0
L
gradients_1/mul_3_grad/Shape_1Shape
ExpandDims*
T0*
out_type0
�
,gradients_1/mul_3_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/mul_3_grad/Shapegradients_1/mul_3_grad/Shape_1*
T0
k
gradients_1/mul_3_grad/MulMul3gradients_1/Minimum_grad/tuple/control_dependency_1
ExpandDims*
T0
�
gradients_1/mul_3_grad/SumSumgradients_1/mul_3_grad/Mul,gradients_1/mul_3_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
z
gradients_1/mul_3_grad/ReshapeReshapegradients_1/mul_3_grad/Sumgradients_1/mul_3_grad/Shape*
T0*
Tshape0
r
gradients_1/mul_3_grad/Mul_1Mulclip_by_value_13gradients_1/Minimum_grad/tuple/control_dependency_1*
T0
�
gradients_1/mul_3_grad/Sum_1Sumgradients_1/mul_3_grad/Mul_1.gradients_1/mul_3_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
�
 gradients_1/mul_3_grad/Reshape_1Reshapegradients_1/mul_3_grad/Sum_1gradients_1/mul_3_grad/Shape_1*
T0*
Tshape0
s
'gradients_1/mul_3_grad/tuple/group_depsNoOp^gradients_1/mul_3_grad/Reshape!^gradients_1/mul_3_grad/Reshape_1
�
/gradients_1/mul_3_grad/tuple/control_dependencyIdentitygradients_1/mul_3_grad/Reshape(^gradients_1/mul_3_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients_1/mul_3_grad/Reshape
�
1gradients_1/mul_3_grad/tuple/control_dependency_1Identity gradients_1/mul_3_grad/Reshape_1(^gradients_1/mul_3_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients_1/mul_3_grad/Reshape_1
@
gradients_1/zeros_like_2	ZerosLikeDynamicPartition*
T0
O
'gradients_1/DynamicPartition_grad/ShapeShapeCast*
T0*
out_type0
U
'gradients_1/DynamicPartition_grad/ConstConst*
valueB: *
dtype0
�
&gradients_1/DynamicPartition_grad/ProdProd'gradients_1/DynamicPartition_grad/Shape'gradients_1/DynamicPartition_grad/Const*

Tidx0*
	keep_dims( *
T0
W
-gradients_1/DynamicPartition_grad/range/startConst*
value	B : *
dtype0
W
-gradients_1/DynamicPartition_grad/range/deltaConst*
value	B :*
dtype0
�
'gradients_1/DynamicPartition_grad/rangeRange-gradients_1/DynamicPartition_grad/range/start&gradients_1/DynamicPartition_grad/Prod-gradients_1/DynamicPartition_grad/range/delta*

Tidx0
�
)gradients_1/DynamicPartition_grad/ReshapeReshape'gradients_1/DynamicPartition_grad/range'gradients_1/DynamicPartition_grad/Shape*
T0*
Tshape0
�
2gradients_1/DynamicPartition_grad/DynamicPartitionDynamicPartition)gradients_1/DynamicPartition_grad/ReshapeCast*
num_partitions*
T0
�
/gradients_1/DynamicPartition_grad/DynamicStitchDynamicStitch2gradients_1/DynamicPartition_grad/DynamicPartition4gradients_1/DynamicPartition_grad/DynamicPartition:1gradients_1/zeros_like_2gradients_1/Mean_1_grad/truediv*
T0*
N
T
)gradients_1/DynamicPartition_grad/Shape_1ShapeMaximum*
T0*
out_type0
�
+gradients_1/DynamicPartition_grad/Reshape_1Reshape/gradients_1/DynamicPartition_grad/DynamicStitch)gradients_1/DynamicPartition_grad/Shape_1*
T0*
Tshape0
�
Bgradients_1/softmax_cross_entropy_with_logits/Reshape_2_grad/ShapeShape!softmax_cross_entropy_with_logits*
T0*
out_type0
�
Dgradients_1/softmax_cross_entropy_with_logits/Reshape_2_grad/ReshapeReshapegradients_1/stack_grad/unstackBgradients_1/softmax_cross_entropy_with_logits/Reshape_2_grad/Shape*
T0*
Tshape0
a
&gradients_1/clip_by_value_1_grad/ShapeShapeclip_by_value_1/Minimum*
T0*
out_type0
Q
(gradients_1/clip_by_value_1_grad/Shape_1Const*
valueB *
dtype0
{
(gradients_1/clip_by_value_1_grad/Shape_2Shape/gradients_1/mul_3_grad/tuple/control_dependency*
T0*
out_type0
Y
,gradients_1/clip_by_value_1_grad/zeros/ConstConst*
valueB
 *    *
dtype0
�
&gradients_1/clip_by_value_1_grad/zerosFill(gradients_1/clip_by_value_1_grad/Shape_2,gradients_1/clip_by_value_1_grad/zeros/Const*
T0*

index_type0
f
-gradients_1/clip_by_value_1_grad/GreaterEqualGreaterEqualclip_by_value_1/Minimumsub_2*
T0
�
6gradients_1/clip_by_value_1_grad/BroadcastGradientArgsBroadcastGradientArgs&gradients_1/clip_by_value_1_grad/Shape(gradients_1/clip_by_value_1_grad/Shape_1*
T0
�
'gradients_1/clip_by_value_1_grad/SelectSelect-gradients_1/clip_by_value_1_grad/GreaterEqual/gradients_1/mul_3_grad/tuple/control_dependency&gradients_1/clip_by_value_1_grad/zeros*
T0
�
$gradients_1/clip_by_value_1_grad/SumSum'gradients_1/clip_by_value_1_grad/Select6gradients_1/clip_by_value_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
�
(gradients_1/clip_by_value_1_grad/ReshapeReshape$gradients_1/clip_by_value_1_grad/Sum&gradients_1/clip_by_value_1_grad/Shape*
T0*
Tshape0
�
)gradients_1/clip_by_value_1_grad/Select_1Select-gradients_1/clip_by_value_1_grad/GreaterEqual&gradients_1/clip_by_value_1_grad/zeros/gradients_1/mul_3_grad/tuple/control_dependency*
T0
�
&gradients_1/clip_by_value_1_grad/Sum_1Sum)gradients_1/clip_by_value_1_grad/Select_18gradients_1/clip_by_value_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
�
*gradients_1/clip_by_value_1_grad/Reshape_1Reshape&gradients_1/clip_by_value_1_grad/Sum_1(gradients_1/clip_by_value_1_grad/Shape_1*
T0*
Tshape0
�
1gradients_1/clip_by_value_1_grad/tuple/group_depsNoOp)^gradients_1/clip_by_value_1_grad/Reshape+^gradients_1/clip_by_value_1_grad/Reshape_1
�
9gradients_1/clip_by_value_1_grad/tuple/control_dependencyIdentity(gradients_1/clip_by_value_1_grad/Reshape2^gradients_1/clip_by_value_1_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients_1/clip_by_value_1_grad/Reshape
�
;gradients_1/clip_by_value_1_grad/tuple/control_dependency_1Identity*gradients_1/clip_by_value_1_grad/Reshape_12^gradients_1/clip_by_value_1_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients_1/clip_by_value_1_grad/Reshape_1
S
gradients_1/Maximum_grad/ShapeShapeSquaredDifference*
T0*
out_type0
W
 gradients_1/Maximum_grad/Shape_1ShapeSquaredDifference_1*
T0*
out_type0
o
 gradients_1/Maximum_grad/Shape_2Shape+gradients_1/DynamicPartition_grad/Reshape_1*
T0*
out_type0
Q
$gradients_1/Maximum_grad/zeros/ConstConst*
valueB
 *    *
dtype0
�
gradients_1/Maximum_grad/zerosFill gradients_1/Maximum_grad/Shape_2$gradients_1/Maximum_grad/zeros/Const*
T0*

index_type0
f
%gradients_1/Maximum_grad/GreaterEqualGreaterEqualSquaredDifferenceSquaredDifference_1*
T0
�
.gradients_1/Maximum_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/Maximum_grad/Shape gradients_1/Maximum_grad/Shape_1*
T0
�
gradients_1/Maximum_grad/SelectSelect%gradients_1/Maximum_grad/GreaterEqual+gradients_1/DynamicPartition_grad/Reshape_1gradients_1/Maximum_grad/zeros*
T0
�
gradients_1/Maximum_grad/SumSumgradients_1/Maximum_grad/Select.gradients_1/Maximum_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
�
 gradients_1/Maximum_grad/ReshapeReshapegradients_1/Maximum_grad/Sumgradients_1/Maximum_grad/Shape*
T0*
Tshape0
�
!gradients_1/Maximum_grad/Select_1Select%gradients_1/Maximum_grad/GreaterEqualgradients_1/Maximum_grad/zeros+gradients_1/DynamicPartition_grad/Reshape_1*
T0
�
gradients_1/Maximum_grad/Sum_1Sum!gradients_1/Maximum_grad/Select_10gradients_1/Maximum_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
�
"gradients_1/Maximum_grad/Reshape_1Reshapegradients_1/Maximum_grad/Sum_1 gradients_1/Maximum_grad/Shape_1*
T0*
Tshape0
y
)gradients_1/Maximum_grad/tuple/group_depsNoOp!^gradients_1/Maximum_grad/Reshape#^gradients_1/Maximum_grad/Reshape_1
�
1gradients_1/Maximum_grad/tuple/control_dependencyIdentity gradients_1/Maximum_grad/Reshape*^gradients_1/Maximum_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients_1/Maximum_grad/Reshape
�
3gradients_1/Maximum_grad/tuple/control_dependency_1Identity"gradients_1/Maximum_grad/Reshape_1*^gradients_1/Maximum_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients_1/Maximum_grad/Reshape_1
S
gradients_1/zeros_like_3	ZerosLike#softmax_cross_entropy_with_logits:1*
T0
t
Agradients_1/softmax_cross_entropy_with_logits_grad/ExpandDims/dimConst*
valueB :
���������*
dtype0
�
=gradients_1/softmax_cross_entropy_with_logits_grad/ExpandDims
ExpandDimsDgradients_1/softmax_cross_entropy_with_logits/Reshape_2_grad/ReshapeAgradients_1/softmax_cross_entropy_with_logits_grad/ExpandDims/dim*

Tdim0*
T0
�
6gradients_1/softmax_cross_entropy_with_logits_grad/mulMul=gradients_1/softmax_cross_entropy_with_logits_grad/ExpandDims#softmax_cross_entropy_with_logits:1*
T0

=gradients_1/softmax_cross_entropy_with_logits_grad/LogSoftmax
LogSoftmax)softmax_cross_entropy_with_logits/Reshape*
T0
�
6gradients_1/softmax_cross_entropy_with_logits_grad/NegNeg=gradients_1/softmax_cross_entropy_with_logits_grad/LogSoftmax*
T0
v
Cgradients_1/softmax_cross_entropy_with_logits_grad/ExpandDims_1/dimConst*
valueB :
���������*
dtype0
�
?gradients_1/softmax_cross_entropy_with_logits_grad/ExpandDims_1
ExpandDimsDgradients_1/softmax_cross_entropy_with_logits/Reshape_2_grad/ReshapeCgradients_1/softmax_cross_entropy_with_logits_grad/ExpandDims_1/dim*

Tdim0*
T0
�
8gradients_1/softmax_cross_entropy_with_logits_grad/mul_1Mul?gradients_1/softmax_cross_entropy_with_logits_grad/ExpandDims_16gradients_1/softmax_cross_entropy_with_logits_grad/Neg*
T0
�
Cgradients_1/softmax_cross_entropy_with_logits_grad/tuple/group_depsNoOp7^gradients_1/softmax_cross_entropy_with_logits_grad/mul9^gradients_1/softmax_cross_entropy_with_logits_grad/mul_1
�
Kgradients_1/softmax_cross_entropy_with_logits_grad/tuple/control_dependencyIdentity6gradients_1/softmax_cross_entropy_with_logits_grad/mulD^gradients_1/softmax_cross_entropy_with_logits_grad/tuple/group_deps*
T0*I
_class?
=;loc:@gradients_1/softmax_cross_entropy_with_logits_grad/mul
�
Mgradients_1/softmax_cross_entropy_with_logits_grad/tuple/control_dependency_1Identity8gradients_1/softmax_cross_entropy_with_logits_grad/mul_1D^gradients_1/softmax_cross_entropy_with_logits_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients_1/softmax_cross_entropy_with_logits_grad/mul_1
U
.gradients_1/clip_by_value_1/Minimum_grad/ShapeShapeExp*
T0*
out_type0
Y
0gradients_1/clip_by_value_1/Minimum_grad/Shape_1Const*
valueB *
dtype0
�
0gradients_1/clip_by_value_1/Minimum_grad/Shape_2Shape9gradients_1/clip_by_value_1_grad/tuple/control_dependency*
T0*
out_type0
a
4gradients_1/clip_by_value_1/Minimum_grad/zeros/ConstConst*
valueB
 *    *
dtype0
�
.gradients_1/clip_by_value_1/Minimum_grad/zerosFill0gradients_1/clip_by_value_1/Minimum_grad/Shape_24gradients_1/clip_by_value_1/Minimum_grad/zeros/Const*
T0*

index_type0
T
2gradients_1/clip_by_value_1/Minimum_grad/LessEqual	LessEqualExpadd_8*
T0
�
>gradients_1/clip_by_value_1/Minimum_grad/BroadcastGradientArgsBroadcastGradientArgs.gradients_1/clip_by_value_1/Minimum_grad/Shape0gradients_1/clip_by_value_1/Minimum_grad/Shape_1*
T0
�
/gradients_1/clip_by_value_1/Minimum_grad/SelectSelect2gradients_1/clip_by_value_1/Minimum_grad/LessEqual9gradients_1/clip_by_value_1_grad/tuple/control_dependency.gradients_1/clip_by_value_1/Minimum_grad/zeros*
T0
�
,gradients_1/clip_by_value_1/Minimum_grad/SumSum/gradients_1/clip_by_value_1/Minimum_grad/Select>gradients_1/clip_by_value_1/Minimum_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
�
0gradients_1/clip_by_value_1/Minimum_grad/ReshapeReshape,gradients_1/clip_by_value_1/Minimum_grad/Sum.gradients_1/clip_by_value_1/Minimum_grad/Shape*
T0*
Tshape0
�
1gradients_1/clip_by_value_1/Minimum_grad/Select_1Select2gradients_1/clip_by_value_1/Minimum_grad/LessEqual.gradients_1/clip_by_value_1/Minimum_grad/zeros9gradients_1/clip_by_value_1_grad/tuple/control_dependency*
T0
�
.gradients_1/clip_by_value_1/Minimum_grad/Sum_1Sum1gradients_1/clip_by_value_1/Minimum_grad/Select_1@gradients_1/clip_by_value_1/Minimum_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
�
2gradients_1/clip_by_value_1/Minimum_grad/Reshape_1Reshape.gradients_1/clip_by_value_1/Minimum_grad/Sum_10gradients_1/clip_by_value_1/Minimum_grad/Shape_1*
T0*
Tshape0
�
9gradients_1/clip_by_value_1/Minimum_grad/tuple/group_depsNoOp1^gradients_1/clip_by_value_1/Minimum_grad/Reshape3^gradients_1/clip_by_value_1/Minimum_grad/Reshape_1
�
Agradients_1/clip_by_value_1/Minimum_grad/tuple/control_dependencyIdentity0gradients_1/clip_by_value_1/Minimum_grad/Reshape:^gradients_1/clip_by_value_1/Minimum_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients_1/clip_by_value_1/Minimum_grad/Reshape
�
Cgradients_1/clip_by_value_1/Minimum_grad/tuple/control_dependency_1Identity2gradients_1/clip_by_value_1/Minimum_grad/Reshape_1:^gradients_1/clip_by_value_1/Minimum_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients_1/clip_by_value_1/Minimum_grad/Reshape_1
�
)gradients_1/SquaredDifference_grad/scalarConst2^gradients_1/Maximum_grad/tuple/control_dependency*
valueB
 *   @*
dtype0
�
&gradients_1/SquaredDifference_grad/MulMul)gradients_1/SquaredDifference_grad/scalar1gradients_1/Maximum_grad/tuple/control_dependency*
T0
�
&gradients_1/SquaredDifference_grad/subSubextrinsic_returnsSum_62^gradients_1/Maximum_grad/tuple/control_dependency*
T0
�
(gradients_1/SquaredDifference_grad/mul_1Mul&gradients_1/SquaredDifference_grad/Mul&gradients_1/SquaredDifference_grad/sub*
T0
]
(gradients_1/SquaredDifference_grad/ShapeShapeextrinsic_returns*
T0*
out_type0
S
*gradients_1/SquaredDifference_grad/Shape_1ShapeSum_6*
T0*
out_type0
�
8gradients_1/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgs(gradients_1/SquaredDifference_grad/Shape*gradients_1/SquaredDifference_grad/Shape_1*
T0
�
&gradients_1/SquaredDifference_grad/SumSum(gradients_1/SquaredDifference_grad/mul_18gradients_1/SquaredDifference_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
�
*gradients_1/SquaredDifference_grad/ReshapeReshape&gradients_1/SquaredDifference_grad/Sum(gradients_1/SquaredDifference_grad/Shape*
T0*
Tshape0
�
(gradients_1/SquaredDifference_grad/Sum_1Sum(gradients_1/SquaredDifference_grad/mul_1:gradients_1/SquaredDifference_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
�
,gradients_1/SquaredDifference_grad/Reshape_1Reshape(gradients_1/SquaredDifference_grad/Sum_1*gradients_1/SquaredDifference_grad/Shape_1*
T0*
Tshape0
d
&gradients_1/SquaredDifference_grad/NegNeg,gradients_1/SquaredDifference_grad/Reshape_1*
T0
�
3gradients_1/SquaredDifference_grad/tuple/group_depsNoOp'^gradients_1/SquaredDifference_grad/Neg+^gradients_1/SquaredDifference_grad/Reshape
�
;gradients_1/SquaredDifference_grad/tuple/control_dependencyIdentity*gradients_1/SquaredDifference_grad/Reshape4^gradients_1/SquaredDifference_grad/tuple/group_deps*
T0*=
_class3
1/loc:@gradients_1/SquaredDifference_grad/Reshape
�
=gradients_1/SquaredDifference_grad/tuple/control_dependency_1Identity&gradients_1/SquaredDifference_grad/Neg4^gradients_1/SquaredDifference_grad/tuple/group_deps*
T0*9
_class/
-+loc:@gradients_1/SquaredDifference_grad/Neg
�
+gradients_1/SquaredDifference_1_grad/scalarConst4^gradients_1/Maximum_grad/tuple/control_dependency_1*
valueB
 *   @*
dtype0
�
(gradients_1/SquaredDifference_1_grad/MulMul+gradients_1/SquaredDifference_1_grad/scalar3gradients_1/Maximum_grad/tuple/control_dependency_1*
T0
�
(gradients_1/SquaredDifference_1_grad/subSubextrinsic_returnsadd_74^gradients_1/Maximum_grad/tuple/control_dependency_1*
T0
�
*gradients_1/SquaredDifference_1_grad/mul_1Mul(gradients_1/SquaredDifference_1_grad/Mul(gradients_1/SquaredDifference_1_grad/sub*
T0
_
*gradients_1/SquaredDifference_1_grad/ShapeShapeextrinsic_returns*
T0*
out_type0
U
,gradients_1/SquaredDifference_1_grad/Shape_1Shapeadd_7*
T0*
out_type0
�
:gradients_1/SquaredDifference_1_grad/BroadcastGradientArgsBroadcastGradientArgs*gradients_1/SquaredDifference_1_grad/Shape,gradients_1/SquaredDifference_1_grad/Shape_1*
T0
�
(gradients_1/SquaredDifference_1_grad/SumSum*gradients_1/SquaredDifference_1_grad/mul_1:gradients_1/SquaredDifference_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
�
,gradients_1/SquaredDifference_1_grad/ReshapeReshape(gradients_1/SquaredDifference_1_grad/Sum*gradients_1/SquaredDifference_1_grad/Shape*
T0*
Tshape0
�
*gradients_1/SquaredDifference_1_grad/Sum_1Sum*gradients_1/SquaredDifference_1_grad/mul_1<gradients_1/SquaredDifference_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
�
.gradients_1/SquaredDifference_1_grad/Reshape_1Reshape*gradients_1/SquaredDifference_1_grad/Sum_1,gradients_1/SquaredDifference_1_grad/Shape_1*
T0*
Tshape0
h
(gradients_1/SquaredDifference_1_grad/NegNeg.gradients_1/SquaredDifference_1_grad/Reshape_1*
T0
�
5gradients_1/SquaredDifference_1_grad/tuple/group_depsNoOp)^gradients_1/SquaredDifference_1_grad/Neg-^gradients_1/SquaredDifference_1_grad/Reshape
�
=gradients_1/SquaredDifference_1_grad/tuple/control_dependencyIdentity,gradients_1/SquaredDifference_1_grad/Reshape6^gradients_1/SquaredDifference_1_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients_1/SquaredDifference_1_grad/Reshape
�
?gradients_1/SquaredDifference_1_grad/tuple/control_dependency_1Identity(gradients_1/SquaredDifference_1_grad/Neg6^gradients_1/SquaredDifference_1_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients_1/SquaredDifference_1_grad/Neg
s
@gradients_1/softmax_cross_entropy_with_logits/Reshape_grad/ShapeShapestrided_slice_8*
T0*
out_type0
�
Bgradients_1/softmax_cross_entropy_with_logits/Reshape_grad/ReshapeReshapeKgradients_1/softmax_cross_entropy_with_logits_grad/tuple/control_dependency@gradients_1/softmax_cross_entropy_with_logits/Reshape_grad/Shape*
T0*
Tshape0
o
Bgradients_1/softmax_cross_entropy_with_logits/Reshape_1_grad/ShapeShape	Softmax_2*
T0*
out_type0
�
Dgradients_1/softmax_cross_entropy_with_logits/Reshape_1_grad/ReshapeReshapeMgradients_1/softmax_cross_entropy_with_logits_grad/tuple/control_dependency_1Bgradients_1/softmax_cross_entropy_with_logits/Reshape_1_grad/Shape*
T0*
Tshape0
�
gradients_1/AddNAddN/gradients_1/mul_2_grad/tuple/control_dependencyAgradients_1/clip_by_value_1/Minimum_grad/tuple/control_dependency*
T0*1
_class'
%#loc:@gradients_1/mul_2_grad/Reshape*
N
?
gradients_1/Exp_grad/mulMulgradients_1/AddNExp*
T0
W
gradients_1/Sum_6_grad/ShapeShapeextrinsic_value/BiasAdd*
T0*
out_type0
v
gradients_1/Sum_6_grad/SizeConst*/
_class%
#!loc:@gradients_1/Sum_6_grad/Shape*
value	B :*
dtype0
�
gradients_1/Sum_6_grad/addAddV2Sum_6/reduction_indicesgradients_1/Sum_6_grad/Size*
T0*/
_class%
#!loc:@gradients_1/Sum_6_grad/Shape
�
gradients_1/Sum_6_grad/modFloorModgradients_1/Sum_6_grad/addgradients_1/Sum_6_grad/Size*
T0*/
_class%
#!loc:@gradients_1/Sum_6_grad/Shape
x
gradients_1/Sum_6_grad/Shape_1Const*/
_class%
#!loc:@gradients_1/Sum_6_grad/Shape*
valueB *
dtype0
}
"gradients_1/Sum_6_grad/range/startConst*/
_class%
#!loc:@gradients_1/Sum_6_grad/Shape*
value	B : *
dtype0
}
"gradients_1/Sum_6_grad/range/deltaConst*/
_class%
#!loc:@gradients_1/Sum_6_grad/Shape*
value	B :*
dtype0
�
gradients_1/Sum_6_grad/rangeRange"gradients_1/Sum_6_grad/range/startgradients_1/Sum_6_grad/Size"gradients_1/Sum_6_grad/range/delta*

Tidx0*/
_class%
#!loc:@gradients_1/Sum_6_grad/Shape
|
!gradients_1/Sum_6_grad/Fill/valueConst*/
_class%
#!loc:@gradients_1/Sum_6_grad/Shape*
value	B :*
dtype0
�
gradients_1/Sum_6_grad/FillFillgradients_1/Sum_6_grad/Shape_1!gradients_1/Sum_6_grad/Fill/value*
T0*/
_class%
#!loc:@gradients_1/Sum_6_grad/Shape*

index_type0
�
$gradients_1/Sum_6_grad/DynamicStitchDynamicStitchgradients_1/Sum_6_grad/rangegradients_1/Sum_6_grad/modgradients_1/Sum_6_grad/Shapegradients_1/Sum_6_grad/Fill*
T0*/
_class%
#!loc:@gradients_1/Sum_6_grad/Shape*
N
{
 gradients_1/Sum_6_grad/Maximum/yConst*/
_class%
#!loc:@gradients_1/Sum_6_grad/Shape*
value	B :*
dtype0
�
gradients_1/Sum_6_grad/MaximumMaximum$gradients_1/Sum_6_grad/DynamicStitch gradients_1/Sum_6_grad/Maximum/y*
T0*/
_class%
#!loc:@gradients_1/Sum_6_grad/Shape
�
gradients_1/Sum_6_grad/floordivFloorDivgradients_1/Sum_6_grad/Shapegradients_1/Sum_6_grad/Maximum*
T0*/
_class%
#!loc:@gradients_1/Sum_6_grad/Shape
�
gradients_1/Sum_6_grad/ReshapeReshape=gradients_1/SquaredDifference_grad/tuple/control_dependency_1$gradients_1/Sum_6_grad/DynamicStitch*
T0*
Tshape0

gradients_1/Sum_6_grad/TileTilegradients_1/Sum_6_grad/Reshapegradients_1/Sum_6_grad/floordiv*

Tmultiples0*
T0
X
gradients_1/add_7_grad/ShapeShapeextrinsic_value_estimate*
T0*
out_type0
O
gradients_1/add_7_grad/Shape_1Shapeclip_by_value*
T0*
out_type0
�
,gradients_1/add_7_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/add_7_grad/Shapegradients_1/add_7_grad/Shape_1*
T0
�
gradients_1/add_7_grad/SumSum?gradients_1/SquaredDifference_1_grad/tuple/control_dependency_1,gradients_1/add_7_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
z
gradients_1/add_7_grad/ReshapeReshapegradients_1/add_7_grad/Sumgradients_1/add_7_grad/Shape*
T0*
Tshape0
�
gradients_1/add_7_grad/Sum_1Sum?gradients_1/SquaredDifference_1_grad/tuple/control_dependency_1.gradients_1/add_7_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
�
 gradients_1/add_7_grad/Reshape_1Reshapegradients_1/add_7_grad/Sum_1gradients_1/add_7_grad/Shape_1*
T0*
Tshape0
s
'gradients_1/add_7_grad/tuple/group_depsNoOp^gradients_1/add_7_grad/Reshape!^gradients_1/add_7_grad/Reshape_1
�
/gradients_1/add_7_grad/tuple/control_dependencyIdentitygradients_1/add_7_grad/Reshape(^gradients_1/add_7_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients_1/add_7_grad/Reshape
�
1gradients_1/add_7_grad/tuple/control_dependency_1Identity gradients_1/add_7_grad/Reshape_1(^gradients_1/add_7_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients_1/add_7_grad/Reshape_1
c
&gradients_1/strided_slice_8_grad/ShapeShapeaction_probs/action_probs*
T0*
out_type0
�
1gradients_1/strided_slice_8_grad/StridedSliceGradStridedSliceGrad&gradients_1/strided_slice_8_grad/Shapestrided_slice_8/stackstrided_slice_8/stack_1strided_slice_8/stack_2Bgradients_1/softmax_cross_entropy_with_logits/Reshape_grad/Reshape*
Index0*
T0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask

gradients_1/Softmax_2_grad/mulMulDgradients_1/softmax_cross_entropy_with_logits/Reshape_1_grad/Reshape	Softmax_2*
T0
c
0gradients_1/Softmax_2_grad/Sum/reduction_indicesConst*
valueB :
���������*
dtype0
�
gradients_1/Softmax_2_grad/SumSumgradients_1/Softmax_2_grad/mul0gradients_1/Softmax_2_grad/Sum/reduction_indices*

Tidx0*
	keep_dims(*
T0
�
gradients_1/Softmax_2_grad/subSubDgradients_1/softmax_cross_entropy_with_logits/Reshape_1_grad/Reshapegradients_1/Softmax_2_grad/Sum*
T0
[
 gradients_1/Softmax_2_grad/mul_1Mulgradients_1/Softmax_2_grad/sub	Softmax_2*
T0
E
gradients_1/sub_1_grad/ShapeShapeSum_3*
T0*
out_type0
G
gradients_1/sub_1_grad/Shape_1ShapeSum_4*
T0*
out_type0
�
,gradients_1/sub_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/sub_1_grad/Shapegradients_1/sub_1_grad/Shape_1*
T0
�
gradients_1/sub_1_grad/SumSumgradients_1/Exp_grad/mul,gradients_1/sub_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
z
gradients_1/sub_1_grad/ReshapeReshapegradients_1/sub_1_grad/Sumgradients_1/sub_1_grad/Shape*
T0*
Tshape0
D
gradients_1/sub_1_grad/NegNeggradients_1/Exp_grad/mul*
T0
�
gradients_1/sub_1_grad/Sum_1Sumgradients_1/sub_1_grad/Neg.gradients_1/sub_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
�
 gradients_1/sub_1_grad/Reshape_1Reshapegradients_1/sub_1_grad/Sum_1gradients_1/sub_1_grad/Shape_1*
T0*
Tshape0
s
'gradients_1/sub_1_grad/tuple/group_depsNoOp^gradients_1/sub_1_grad/Reshape!^gradients_1/sub_1_grad/Reshape_1
�
/gradients_1/sub_1_grad/tuple/control_dependencyIdentitygradients_1/sub_1_grad/Reshape(^gradients_1/sub_1_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients_1/sub_1_grad/Reshape
�
1gradients_1/sub_1_grad/tuple/control_dependency_1Identity gradients_1/sub_1_grad/Reshape_1(^gradients_1/sub_1_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients_1/sub_1_grad/Reshape_1
]
$gradients_1/clip_by_value_grad/ShapeShapeclip_by_value/Minimum*
T0*
out_type0
O
&gradients_1/clip_by_value_grad/Shape_1Const*
valueB *
dtype0
{
&gradients_1/clip_by_value_grad/Shape_2Shape1gradients_1/add_7_grad/tuple/control_dependency_1*
T0*
out_type0
W
*gradients_1/clip_by_value_grad/zeros/ConstConst*
valueB
 *    *
dtype0
�
$gradients_1/clip_by_value_grad/zerosFill&gradients_1/clip_by_value_grad/Shape_2*gradients_1/clip_by_value_grad/zeros/Const*
T0*

index_type0
b
+gradients_1/clip_by_value_grad/GreaterEqualGreaterEqualclip_by_value/MinimumNeg_2*
T0
�
4gradients_1/clip_by_value_grad/BroadcastGradientArgsBroadcastGradientArgs$gradients_1/clip_by_value_grad/Shape&gradients_1/clip_by_value_grad/Shape_1*
T0
�
%gradients_1/clip_by_value_grad/SelectSelect+gradients_1/clip_by_value_grad/GreaterEqual1gradients_1/add_7_grad/tuple/control_dependency_1$gradients_1/clip_by_value_grad/zeros*
T0
�
"gradients_1/clip_by_value_grad/SumSum%gradients_1/clip_by_value_grad/Select4gradients_1/clip_by_value_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
�
&gradients_1/clip_by_value_grad/ReshapeReshape"gradients_1/clip_by_value_grad/Sum$gradients_1/clip_by_value_grad/Shape*
T0*
Tshape0
�
'gradients_1/clip_by_value_grad/Select_1Select+gradients_1/clip_by_value_grad/GreaterEqual$gradients_1/clip_by_value_grad/zeros1gradients_1/add_7_grad/tuple/control_dependency_1*
T0
�
$gradients_1/clip_by_value_grad/Sum_1Sum'gradients_1/clip_by_value_grad/Select_16gradients_1/clip_by_value_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
�
(gradients_1/clip_by_value_grad/Reshape_1Reshape$gradients_1/clip_by_value_grad/Sum_1&gradients_1/clip_by_value_grad/Shape_1*
T0*
Tshape0
�
/gradients_1/clip_by_value_grad/tuple/group_depsNoOp'^gradients_1/clip_by_value_grad/Reshape)^gradients_1/clip_by_value_grad/Reshape_1
�
7gradients_1/clip_by_value_grad/tuple/control_dependencyIdentity&gradients_1/clip_by_value_grad/Reshape0^gradients_1/clip_by_value_grad/tuple/group_deps*
T0*9
_class/
-+loc:@gradients_1/clip_by_value_grad/Reshape
�
9gradients_1/clip_by_value_grad/tuple/control_dependency_1Identity(gradients_1/clip_by_value_grad/Reshape_10^gradients_1/clip_by_value_grad/tuple/group_deps*
T0*;
_class1
/-loc:@gradients_1/clip_by_value_grad/Reshape_1
c
&gradients_1/strided_slice_7_grad/ShapeShapeaction_probs/action_probs*
T0*
out_type0
�
1gradients_1/strided_slice_7_grad/StridedSliceGradStridedSliceGrad&gradients_1/strided_slice_7_grad/Shapestrided_slice_7/stackstrided_slice_7/stack_1strided_slice_7/stack_2 gradients_1/Softmax_2_grad/mul_1*
Index0*
T0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask
G
gradients_1/Sum_3_grad/ShapeShapestack_1*
T0*
out_type0
v
gradients_1/Sum_3_grad/SizeConst*/
_class%
#!loc:@gradients_1/Sum_3_grad/Shape*
value	B :*
dtype0
�
gradients_1/Sum_3_grad/addAddV2Sum_3/reduction_indicesgradients_1/Sum_3_grad/Size*
T0*/
_class%
#!loc:@gradients_1/Sum_3_grad/Shape
�
gradients_1/Sum_3_grad/modFloorModgradients_1/Sum_3_grad/addgradients_1/Sum_3_grad/Size*
T0*/
_class%
#!loc:@gradients_1/Sum_3_grad/Shape
x
gradients_1/Sum_3_grad/Shape_1Const*/
_class%
#!loc:@gradients_1/Sum_3_grad/Shape*
valueB *
dtype0
}
"gradients_1/Sum_3_grad/range/startConst*/
_class%
#!loc:@gradients_1/Sum_3_grad/Shape*
value	B : *
dtype0
}
"gradients_1/Sum_3_grad/range/deltaConst*/
_class%
#!loc:@gradients_1/Sum_3_grad/Shape*
value	B :*
dtype0
�
gradients_1/Sum_3_grad/rangeRange"gradients_1/Sum_3_grad/range/startgradients_1/Sum_3_grad/Size"gradients_1/Sum_3_grad/range/delta*

Tidx0*/
_class%
#!loc:@gradients_1/Sum_3_grad/Shape
|
!gradients_1/Sum_3_grad/Fill/valueConst*/
_class%
#!loc:@gradients_1/Sum_3_grad/Shape*
value	B :*
dtype0
�
gradients_1/Sum_3_grad/FillFillgradients_1/Sum_3_grad/Shape_1!gradients_1/Sum_3_grad/Fill/value*
T0*/
_class%
#!loc:@gradients_1/Sum_3_grad/Shape*

index_type0
�
$gradients_1/Sum_3_grad/DynamicStitchDynamicStitchgradients_1/Sum_3_grad/rangegradients_1/Sum_3_grad/modgradients_1/Sum_3_grad/Shapegradients_1/Sum_3_grad/Fill*
T0*/
_class%
#!loc:@gradients_1/Sum_3_grad/Shape*
N
{
 gradients_1/Sum_3_grad/Maximum/yConst*/
_class%
#!loc:@gradients_1/Sum_3_grad/Shape*
value	B :*
dtype0
�
gradients_1/Sum_3_grad/MaximumMaximum$gradients_1/Sum_3_grad/DynamicStitch gradients_1/Sum_3_grad/Maximum/y*
T0*/
_class%
#!loc:@gradients_1/Sum_3_grad/Shape
�
gradients_1/Sum_3_grad/floordivFloorDivgradients_1/Sum_3_grad/Shapegradients_1/Sum_3_grad/Maximum*
T0*/
_class%
#!loc:@gradients_1/Sum_3_grad/Shape
�
gradients_1/Sum_3_grad/ReshapeReshape/gradients_1/sub_1_grad/tuple/control_dependency$gradients_1/Sum_3_grad/DynamicStitch*
T0*
Tshape0

gradients_1/Sum_3_grad/TileTilegradients_1/Sum_3_grad/Reshapegradients_1/Sum_3_grad/floordiv*

Tmultiples0*
T0
S
,gradients_1/clip_by_value/Minimum_grad/ShapeShapesub*
T0*
out_type0
W
.gradients_1/clip_by_value/Minimum_grad/Shape_1Const*
valueB *
dtype0
�
.gradients_1/clip_by_value/Minimum_grad/Shape_2Shape7gradients_1/clip_by_value_grad/tuple/control_dependency*
T0*
out_type0
_
2gradients_1/clip_by_value/Minimum_grad/zeros/ConstConst*
valueB
 *    *
dtype0
�
,gradients_1/clip_by_value/Minimum_grad/zerosFill.gradients_1/clip_by_value/Minimum_grad/Shape_22gradients_1/clip_by_value/Minimum_grad/zeros/Const*
T0*

index_type0
^
0gradients_1/clip_by_value/Minimum_grad/LessEqual	LessEqualsubPolynomialDecay_1*
T0
�
<gradients_1/clip_by_value/Minimum_grad/BroadcastGradientArgsBroadcastGradientArgs,gradients_1/clip_by_value/Minimum_grad/Shape.gradients_1/clip_by_value/Minimum_grad/Shape_1*
T0
�
-gradients_1/clip_by_value/Minimum_grad/SelectSelect0gradients_1/clip_by_value/Minimum_grad/LessEqual7gradients_1/clip_by_value_grad/tuple/control_dependency,gradients_1/clip_by_value/Minimum_grad/zeros*
T0
�
*gradients_1/clip_by_value/Minimum_grad/SumSum-gradients_1/clip_by_value/Minimum_grad/Select<gradients_1/clip_by_value/Minimum_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
�
.gradients_1/clip_by_value/Minimum_grad/ReshapeReshape*gradients_1/clip_by_value/Minimum_grad/Sum,gradients_1/clip_by_value/Minimum_grad/Shape*
T0*
Tshape0
�
/gradients_1/clip_by_value/Minimum_grad/Select_1Select0gradients_1/clip_by_value/Minimum_grad/LessEqual,gradients_1/clip_by_value/Minimum_grad/zeros7gradients_1/clip_by_value_grad/tuple/control_dependency*
T0
�
,gradients_1/clip_by_value/Minimum_grad/Sum_1Sum/gradients_1/clip_by_value/Minimum_grad/Select_1>gradients_1/clip_by_value/Minimum_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
�
0gradients_1/clip_by_value/Minimum_grad/Reshape_1Reshape,gradients_1/clip_by_value/Minimum_grad/Sum_1.gradients_1/clip_by_value/Minimum_grad/Shape_1*
T0*
Tshape0
�
7gradients_1/clip_by_value/Minimum_grad/tuple/group_depsNoOp/^gradients_1/clip_by_value/Minimum_grad/Reshape1^gradients_1/clip_by_value/Minimum_grad/Reshape_1
�
?gradients_1/clip_by_value/Minimum_grad/tuple/control_dependencyIdentity.gradients_1/clip_by_value/Minimum_grad/Reshape8^gradients_1/clip_by_value/Minimum_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_1/clip_by_value/Minimum_grad/Reshape
�
Agradients_1/clip_by_value/Minimum_grad/tuple/control_dependency_1Identity0gradients_1/clip_by_value/Minimum_grad/Reshape_18^gradients_1/clip_by_value/Minimum_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients_1/clip_by_value/Minimum_grad/Reshape_1
g
 gradients_1/stack_1_grad/unstackUnpackgradients_1/Sum_3_grad/Tile*
T0*	
num*

axis
C
gradients_1/sub_grad/ShapeShapeSum_5*
T0*
out_type0
X
gradients_1/sub_grad/Shape_1Shapeextrinsic_value_estimate*
T0*
out_type0
�
*gradients_1/sub_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/sub_grad/Shapegradients_1/sub_grad/Shape_1*
T0
�
gradients_1/sub_grad/SumSum?gradients_1/clip_by_value/Minimum_grad/tuple/control_dependency*gradients_1/sub_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
t
gradients_1/sub_grad/ReshapeReshapegradients_1/sub_grad/Sumgradients_1/sub_grad/Shape*
T0*
Tshape0
i
gradients_1/sub_grad/NegNeg?gradients_1/clip_by_value/Minimum_grad/tuple/control_dependency*
T0
�
gradients_1/sub_grad/Sum_1Sumgradients_1/sub_grad/Neg,gradients_1/sub_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
z
gradients_1/sub_grad/Reshape_1Reshapegradients_1/sub_grad/Sum_1gradients_1/sub_grad/Shape_1*
T0*
Tshape0
m
%gradients_1/sub_grad/tuple/group_depsNoOp^gradients_1/sub_grad/Reshape^gradients_1/sub_grad/Reshape_1
�
-gradients_1/sub_grad/tuple/control_dependencyIdentitygradients_1/sub_grad/Reshape&^gradients_1/sub_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients_1/sub_grad/Reshape
�
/gradients_1/sub_grad/tuple/control_dependency_1Identitygradients_1/sub_grad/Reshape_1&^gradients_1/sub_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients_1/sub_grad/Reshape_1
J
gradients_1/Neg_grad/NegNeg gradients_1/stack_1_grad/unstack*
T0
W
gradients_1/Sum_5_grad/ShapeShapeextrinsic_value/BiasAdd*
T0*
out_type0
v
gradients_1/Sum_5_grad/SizeConst*/
_class%
#!loc:@gradients_1/Sum_5_grad/Shape*
value	B :*
dtype0
�
gradients_1/Sum_5_grad/addAddV2Sum_5/reduction_indicesgradients_1/Sum_5_grad/Size*
T0*/
_class%
#!loc:@gradients_1/Sum_5_grad/Shape
�
gradients_1/Sum_5_grad/modFloorModgradients_1/Sum_5_grad/addgradients_1/Sum_5_grad/Size*
T0*/
_class%
#!loc:@gradients_1/Sum_5_grad/Shape
x
gradients_1/Sum_5_grad/Shape_1Const*/
_class%
#!loc:@gradients_1/Sum_5_grad/Shape*
valueB *
dtype0
}
"gradients_1/Sum_5_grad/range/startConst*/
_class%
#!loc:@gradients_1/Sum_5_grad/Shape*
value	B : *
dtype0
}
"gradients_1/Sum_5_grad/range/deltaConst*/
_class%
#!loc:@gradients_1/Sum_5_grad/Shape*
value	B :*
dtype0
�
gradients_1/Sum_5_grad/rangeRange"gradients_1/Sum_5_grad/range/startgradients_1/Sum_5_grad/Size"gradients_1/Sum_5_grad/range/delta*

Tidx0*/
_class%
#!loc:@gradients_1/Sum_5_grad/Shape
|
!gradients_1/Sum_5_grad/Fill/valueConst*/
_class%
#!loc:@gradients_1/Sum_5_grad/Shape*
value	B :*
dtype0
�
gradients_1/Sum_5_grad/FillFillgradients_1/Sum_5_grad/Shape_1!gradients_1/Sum_5_grad/Fill/value*
T0*/
_class%
#!loc:@gradients_1/Sum_5_grad/Shape*

index_type0
�
$gradients_1/Sum_5_grad/DynamicStitchDynamicStitchgradients_1/Sum_5_grad/rangegradients_1/Sum_5_grad/modgradients_1/Sum_5_grad/Shapegradients_1/Sum_5_grad/Fill*
T0*/
_class%
#!loc:@gradients_1/Sum_5_grad/Shape*
N
{
 gradients_1/Sum_5_grad/Maximum/yConst*/
_class%
#!loc:@gradients_1/Sum_5_grad/Shape*
value	B :*
dtype0
�
gradients_1/Sum_5_grad/MaximumMaximum$gradients_1/Sum_5_grad/DynamicStitch gradients_1/Sum_5_grad/Maximum/y*
T0*/
_class%
#!loc:@gradients_1/Sum_5_grad/Shape
�
gradients_1/Sum_5_grad/floordivFloorDivgradients_1/Sum_5_grad/Shapegradients_1/Sum_5_grad/Maximum*
T0*/
_class%
#!loc:@gradients_1/Sum_5_grad/Shape
�
gradients_1/Sum_5_grad/ReshapeReshape-gradients_1/sub_grad/tuple/control_dependency$gradients_1/Sum_5_grad/DynamicStitch*
T0*
Tshape0

gradients_1/Sum_5_grad/TileTilegradients_1/Sum_5_grad/Reshapegradients_1/Sum_5_grad/floordiv*

Tmultiples0*
T0
�
Dgradients_1/softmax_cross_entropy_with_logits_1/Reshape_2_grad/ShapeShape#softmax_cross_entropy_with_logits_1*
T0*
out_type0
�
Fgradients_1/softmax_cross_entropy_with_logits_1/Reshape_2_grad/ReshapeReshapegradients_1/Neg_grad/NegDgradients_1/softmax_cross_entropy_with_logits_1/Reshape_2_grad/Shape*
T0*
Tshape0
�
gradients_1/AddN_1AddNgradients_1/Sum_6_grad/Tilegradients_1/Sum_5_grad/Tile*
T0*.
_class$
" loc:@gradients_1/Sum_6_grad/Tile*
N
w
4gradients_1/extrinsic_value/BiasAdd_grad/BiasAddGradBiasAddGradgradients_1/AddN_1*
T0*
data_formatNHWC
�
9gradients_1/extrinsic_value/BiasAdd_grad/tuple/group_depsNoOp^gradients_1/AddN_15^gradients_1/extrinsic_value/BiasAdd_grad/BiasAddGrad
�
Agradients_1/extrinsic_value/BiasAdd_grad/tuple/control_dependencyIdentitygradients_1/AddN_1:^gradients_1/extrinsic_value/BiasAdd_grad/tuple/group_deps*
T0*.
_class$
" loc:@gradients_1/Sum_6_grad/Tile
�
Cgradients_1/extrinsic_value/BiasAdd_grad/tuple/control_dependency_1Identity4gradients_1/extrinsic_value/BiasAdd_grad/BiasAddGrad:^gradients_1/extrinsic_value/BiasAdd_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients_1/extrinsic_value/BiasAdd_grad/BiasAddGrad
U
gradients_1/zeros_like_4	ZerosLike%softmax_cross_entropy_with_logits_1:1*
T0
v
Cgradients_1/softmax_cross_entropy_with_logits_1_grad/ExpandDims/dimConst*
valueB :
���������*
dtype0
�
?gradients_1/softmax_cross_entropy_with_logits_1_grad/ExpandDims
ExpandDimsFgradients_1/softmax_cross_entropy_with_logits_1/Reshape_2_grad/ReshapeCgradients_1/softmax_cross_entropy_with_logits_1_grad/ExpandDims/dim*

Tdim0*
T0
�
8gradients_1/softmax_cross_entropy_with_logits_1_grad/mulMul?gradients_1/softmax_cross_entropy_with_logits_1_grad/ExpandDims%softmax_cross_entropy_with_logits_1:1*
T0
�
?gradients_1/softmax_cross_entropy_with_logits_1_grad/LogSoftmax
LogSoftmax+softmax_cross_entropy_with_logits_1/Reshape*
T0
�
8gradients_1/softmax_cross_entropy_with_logits_1_grad/NegNeg?gradients_1/softmax_cross_entropy_with_logits_1_grad/LogSoftmax*
T0
x
Egradients_1/softmax_cross_entropy_with_logits_1_grad/ExpandDims_1/dimConst*
valueB :
���������*
dtype0
�
Agradients_1/softmax_cross_entropy_with_logits_1_grad/ExpandDims_1
ExpandDimsFgradients_1/softmax_cross_entropy_with_logits_1/Reshape_2_grad/ReshapeEgradients_1/softmax_cross_entropy_with_logits_1_grad/ExpandDims_1/dim*

Tdim0*
T0
�
:gradients_1/softmax_cross_entropy_with_logits_1_grad/mul_1MulAgradients_1/softmax_cross_entropy_with_logits_1_grad/ExpandDims_18gradients_1/softmax_cross_entropy_with_logits_1_grad/Neg*
T0
�
Egradients_1/softmax_cross_entropy_with_logits_1_grad/tuple/group_depsNoOp9^gradients_1/softmax_cross_entropy_with_logits_1_grad/mul;^gradients_1/softmax_cross_entropy_with_logits_1_grad/mul_1
�
Mgradients_1/softmax_cross_entropy_with_logits_1_grad/tuple/control_dependencyIdentity8gradients_1/softmax_cross_entropy_with_logits_1_grad/mulF^gradients_1/softmax_cross_entropy_with_logits_1_grad/tuple/group_deps*
T0*K
_classA
?=loc:@gradients_1/softmax_cross_entropy_with_logits_1_grad/mul
�
Ogradients_1/softmax_cross_entropy_with_logits_1_grad/tuple/control_dependency_1Identity:gradients_1/softmax_cross_entropy_with_logits_1_grad/mul_1F^gradients_1/softmax_cross_entropy_with_logits_1_grad/tuple/group_deps*
T0*M
_classC
A?loc:@gradients_1/softmax_cross_entropy_with_logits_1_grad/mul_1
�
.gradients_1/extrinsic_value/MatMul_grad/MatMulMatMulAgradients_1/extrinsic_value/BiasAdd_grad/tuple/control_dependencyextrinsic_value/kernel/read*
transpose_b(*
T0*
transpose_a( 
�
0gradients_1/extrinsic_value/MatMul_grad/MatMul_1MatMul	Reshape_2Agradients_1/extrinsic_value/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(
�
8gradients_1/extrinsic_value/MatMul_grad/tuple/group_depsNoOp/^gradients_1/extrinsic_value/MatMul_grad/MatMul1^gradients_1/extrinsic_value/MatMul_grad/MatMul_1
�
@gradients_1/extrinsic_value/MatMul_grad/tuple/control_dependencyIdentity.gradients_1/extrinsic_value/MatMul_grad/MatMul9^gradients_1/extrinsic_value/MatMul_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_1/extrinsic_value/MatMul_grad/MatMul
�
Bgradients_1/extrinsic_value/MatMul_grad/tuple/control_dependency_1Identity0gradients_1/extrinsic_value/MatMul_grad/MatMul_19^gradients_1/extrinsic_value/MatMul_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients_1/extrinsic_value/MatMul_grad/MatMul_1
v
Bgradients_1/softmax_cross_entropy_with_logits_1/Reshape_grad/ShapeShapestrided_slice_10*
T0*
out_type0
�
Dgradients_1/softmax_cross_entropy_with_logits_1/Reshape_grad/ReshapeReshapeMgradients_1/softmax_cross_entropy_with_logits_1_grad/tuple/control_dependencyBgradients_1/softmax_cross_entropy_with_logits_1/Reshape_grad/Shape*
T0*
Tshape0
Z
'gradients_1/strided_slice_10_grad/ShapeShapeconcat_6/concat*
T0*
out_type0
�
2gradients_1/strided_slice_10_grad/StridedSliceGradStridedSliceGrad'gradients_1/strided_slice_10_grad/Shapestrided_slice_10/stackstrided_slice_10/stack_1strided_slice_10/stack_2Dgradients_1/softmax_cross_entropy_with_logits_1/Reshape_grad/Reshape*
Index0*
T0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask
t
!gradients_1/Log_1_grad/Reciprocal
Reciprocaladd_33^gradients_1/strided_slice_10_grad/StridedSliceGrad*
T0
�
gradients_1/Log_1_grad/mulMul2gradients_1/strided_slice_10_grad/StridedSliceGrad!gradients_1/Log_1_grad/Reciprocal*
T0
G
gradients_1/add_3_grad/ShapeShapetruediv*
T0*
out_type0
I
gradients_1/add_3_grad/Shape_1Shapeadd_3/y*
T0*
out_type0
�
,gradients_1/add_3_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/add_3_grad/Shapegradients_1/add_3_grad/Shape_1*
T0
�
gradients_1/add_3_grad/SumSumgradients_1/Log_1_grad/mul,gradients_1/add_3_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
z
gradients_1/add_3_grad/ReshapeReshapegradients_1/add_3_grad/Sumgradients_1/add_3_grad/Shape*
T0*
Tshape0
�
gradients_1/add_3_grad/Sum_1Sumgradients_1/Log_1_grad/mul.gradients_1/add_3_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
�
 gradients_1/add_3_grad/Reshape_1Reshapegradients_1/add_3_grad/Sum_1gradients_1/add_3_grad/Shape_1*
T0*
Tshape0
s
'gradients_1/add_3_grad/tuple/group_depsNoOp^gradients_1/add_3_grad/Reshape!^gradients_1/add_3_grad/Reshape_1
�
/gradients_1/add_3_grad/tuple/control_dependencyIdentitygradients_1/add_3_grad/Reshape(^gradients_1/add_3_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients_1/add_3_grad/Reshape
�
1gradients_1/add_3_grad/tuple/control_dependency_1Identity gradients_1/add_3_grad/Reshape_1(^gradients_1/add_3_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients_1/add_3_grad/Reshape_1
E
gradients_1/truediv_grad/ShapeShapeMul*
T0*
out_type0
G
 gradients_1/truediv_grad/Shape_1ShapeSum*
T0*
out_type0
�
.gradients_1/truediv_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/truediv_grad/Shape gradients_1/truediv_grad/Shape_1*
T0
j
 gradients_1/truediv_grad/RealDivRealDiv/gradients_1/add_3_grad/tuple/control_dependencySum*
T0
�
gradients_1/truediv_grad/SumSum gradients_1/truediv_grad/RealDiv.gradients_1/truediv_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
�
 gradients_1/truediv_grad/ReshapeReshapegradients_1/truediv_grad/Sumgradients_1/truediv_grad/Shape*
T0*
Tshape0
1
gradients_1/truediv_grad/NegNegMul*
T0
Y
"gradients_1/truediv_grad/RealDiv_1RealDivgradients_1/truediv_grad/NegSum*
T0
_
"gradients_1/truediv_grad/RealDiv_2RealDiv"gradients_1/truediv_grad/RealDiv_1Sum*
T0
�
gradients_1/truediv_grad/mulMul/gradients_1/add_3_grad/tuple/control_dependency"gradients_1/truediv_grad/RealDiv_2*
T0
�
gradients_1/truediv_grad/Sum_1Sumgradients_1/truediv_grad/mul0gradients_1/truediv_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
�
"gradients_1/truediv_grad/Reshape_1Reshapegradients_1/truediv_grad/Sum_1 gradients_1/truediv_grad/Shape_1*
T0*
Tshape0
y
)gradients_1/truediv_grad/tuple/group_depsNoOp!^gradients_1/truediv_grad/Reshape#^gradients_1/truediv_grad/Reshape_1
�
1gradients_1/truediv_grad/tuple/control_dependencyIdentity gradients_1/truediv_grad/Reshape*^gradients_1/truediv_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients_1/truediv_grad/Reshape
�
3gradients_1/truediv_grad/tuple/control_dependency_1Identity"gradients_1/truediv_grad/Reshape_1*^gradients_1/truediv_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients_1/truediv_grad/Reshape_1
A
gradients_1/Sum_grad/ShapeShapeMul*
T0*
out_type0
r
gradients_1/Sum_grad/SizeConst*-
_class#
!loc:@gradients_1/Sum_grad/Shape*
value	B :*
dtype0
�
gradients_1/Sum_grad/addAddV2Sum/reduction_indicesgradients_1/Sum_grad/Size*
T0*-
_class#
!loc:@gradients_1/Sum_grad/Shape
�
gradients_1/Sum_grad/modFloorModgradients_1/Sum_grad/addgradients_1/Sum_grad/Size*
T0*-
_class#
!loc:@gradients_1/Sum_grad/Shape
t
gradients_1/Sum_grad/Shape_1Const*-
_class#
!loc:@gradients_1/Sum_grad/Shape*
valueB *
dtype0
y
 gradients_1/Sum_grad/range/startConst*-
_class#
!loc:@gradients_1/Sum_grad/Shape*
value	B : *
dtype0
y
 gradients_1/Sum_grad/range/deltaConst*-
_class#
!loc:@gradients_1/Sum_grad/Shape*
value	B :*
dtype0
�
gradients_1/Sum_grad/rangeRange gradients_1/Sum_grad/range/startgradients_1/Sum_grad/Size gradients_1/Sum_grad/range/delta*

Tidx0*-
_class#
!loc:@gradients_1/Sum_grad/Shape
x
gradients_1/Sum_grad/Fill/valueConst*-
_class#
!loc:@gradients_1/Sum_grad/Shape*
value	B :*
dtype0
�
gradients_1/Sum_grad/FillFillgradients_1/Sum_grad/Shape_1gradients_1/Sum_grad/Fill/value*
T0*-
_class#
!loc:@gradients_1/Sum_grad/Shape*

index_type0
�
"gradients_1/Sum_grad/DynamicStitchDynamicStitchgradients_1/Sum_grad/rangegradients_1/Sum_grad/modgradients_1/Sum_grad/Shapegradients_1/Sum_grad/Fill*
T0*-
_class#
!loc:@gradients_1/Sum_grad/Shape*
N
w
gradients_1/Sum_grad/Maximum/yConst*-
_class#
!loc:@gradients_1/Sum_grad/Shape*
value	B :*
dtype0
�
gradients_1/Sum_grad/MaximumMaximum"gradients_1/Sum_grad/DynamicStitchgradients_1/Sum_grad/Maximum/y*
T0*-
_class#
!loc:@gradients_1/Sum_grad/Shape
�
gradients_1/Sum_grad/floordivFloorDivgradients_1/Sum_grad/Shapegradients_1/Sum_grad/Maximum*
T0*-
_class#
!loc:@gradients_1/Sum_grad/Shape
�
gradients_1/Sum_grad/ReshapeReshape3gradients_1/truediv_grad/tuple/control_dependency_1"gradients_1/Sum_grad/DynamicStitch*
T0*
Tshape0
y
gradients_1/Sum_grad/TileTilegradients_1/Sum_grad/Reshapegradients_1/Sum_grad/floordiv*

Tmultiples0*
T0
�
gradients_1/AddN_2AddN1gradients_1/truediv_grad/tuple/control_dependencygradients_1/Sum_grad/Tile*
T0*3
_class)
'%loc:@gradients_1/truediv_grad/Reshape*
N
C
gradients_1/Mul_grad/ShapeShapeadd_1*
T0*
out_type0
O
gradients_1/Mul_grad/Shape_1Shapestrided_slice_3*
T0*
out_type0
�
*gradients_1/Mul_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/Mul_grad/Shapegradients_1/Mul_grad/Shape_1*
T0
M
gradients_1/Mul_grad/MulMulgradients_1/AddN_2strided_slice_3*
T0
�
gradients_1/Mul_grad/SumSumgradients_1/Mul_grad/Mul*gradients_1/Mul_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
t
gradients_1/Mul_grad/ReshapeReshapegradients_1/Mul_grad/Sumgradients_1/Mul_grad/Shape*
T0*
Tshape0
E
gradients_1/Mul_grad/Mul_1Muladd_1gradients_1/AddN_2*
T0
�
gradients_1/Mul_grad/Sum_1Sumgradients_1/Mul_grad/Mul_1,gradients_1/Mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
z
gradients_1/Mul_grad/Reshape_1Reshapegradients_1/Mul_grad/Sum_1gradients_1/Mul_grad/Shape_1*
T0*
Tshape0
m
%gradients_1/Mul_grad/tuple/group_depsNoOp^gradients_1/Mul_grad/Reshape^gradients_1/Mul_grad/Reshape_1
�
-gradients_1/Mul_grad/tuple/control_dependencyIdentitygradients_1/Mul_grad/Reshape&^gradients_1/Mul_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients_1/Mul_grad/Reshape
�
/gradients_1/Mul_grad/tuple/control_dependency_1Identitygradients_1/Mul_grad/Reshape_1&^gradients_1/Mul_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients_1/Mul_grad/Reshape_1
G
gradients_1/add_1_grad/ShapeShapeSoftmax*
T0*
out_type0
I
gradients_1/add_1_grad/Shape_1Shapeadd_1/y*
T0*
out_type0
�
,gradients_1/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/add_1_grad/Shapegradients_1/add_1_grad/Shape_1*
T0
�
gradients_1/add_1_grad/SumSum-gradients_1/Mul_grad/tuple/control_dependency,gradients_1/add_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
z
gradients_1/add_1_grad/ReshapeReshapegradients_1/add_1_grad/Sumgradients_1/add_1_grad/Shape*
T0*
Tshape0
�
gradients_1/add_1_grad/Sum_1Sum-gradients_1/Mul_grad/tuple/control_dependency.gradients_1/add_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
�
 gradients_1/add_1_grad/Reshape_1Reshapegradients_1/add_1_grad/Sum_1gradients_1/add_1_grad/Shape_1*
T0*
Tshape0
s
'gradients_1/add_1_grad/tuple/group_depsNoOp^gradients_1/add_1_grad/Reshape!^gradients_1/add_1_grad/Reshape_1
�
/gradients_1/add_1_grad/tuple/control_dependencyIdentitygradients_1/add_1_grad/Reshape(^gradients_1/add_1_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients_1/add_1_grad/Reshape
�
1gradients_1/add_1_grad/tuple/control_dependency_1Identity gradients_1/add_1_grad/Reshape_1(^gradients_1/add_1_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients_1/add_1_grad/Reshape_1
f
gradients_1/Softmax_grad/mulMul/gradients_1/add_1_grad/tuple/control_dependencySoftmax*
T0
a
.gradients_1/Softmax_grad/Sum/reduction_indicesConst*
valueB :
���������*
dtype0
�
gradients_1/Softmax_grad/SumSumgradients_1/Softmax_grad/mul.gradients_1/Softmax_grad/Sum/reduction_indices*

Tidx0*
	keep_dims(*
T0
{
gradients_1/Softmax_grad/subSub/gradients_1/add_1_grad/tuple/control_dependencygradients_1/Softmax_grad/Sum*
T0
U
gradients_1/Softmax_grad/mul_1Mulgradients_1/Softmax_grad/subSoftmax*
T0
c
&gradients_1/strided_slice_2_grad/ShapeShapeaction_probs/action_probs*
T0*
out_type0
�
1gradients_1/strided_slice_2_grad/StridedSliceGradStridedSliceGrad&gradients_1/strided_slice_2_grad/Shapestrided_slice_2/stackstrided_slice_2/stack_1strided_slice_2/stack_2gradients_1/Softmax_grad/mul_1*
Index0*
T0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask
�
gradients_1/AddN_3AddN1gradients_1/strided_slice_8_grad/StridedSliceGrad1gradients_1/strided_slice_7_grad/StridedSliceGrad1gradients_1/strided_slice_2_grad/StridedSliceGrad*
T0*D
_class:
86loc:@gradients_1/strided_slice_8_grad/StridedSliceGrad*
N
�
$gradients_1/dense/MatMul_grad/MatMulMatMulgradients_1/AddN_3dense/kernel/read*
transpose_b(*
T0*
transpose_a( 
~
&gradients_1/dense/MatMul_grad/MatMul_1MatMul	Reshape_2gradients_1/AddN_3*
transpose_b( *
T0*
transpose_a(
�
.gradients_1/dense/MatMul_grad/tuple/group_depsNoOp%^gradients_1/dense/MatMul_grad/MatMul'^gradients_1/dense/MatMul_grad/MatMul_1
�
6gradients_1/dense/MatMul_grad/tuple/control_dependencyIdentity$gradients_1/dense/MatMul_grad/MatMul/^gradients_1/dense/MatMul_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients_1/dense/MatMul_grad/MatMul
�
8gradients_1/dense/MatMul_grad/tuple/control_dependency_1Identity&gradients_1/dense/MatMul_grad/MatMul_1/^gradients_1/dense/MatMul_grad/tuple/group_deps*
T0*9
_class/
-+loc:@gradients_1/dense/MatMul_grad/MatMul_1
�
gradients_1/AddN_4AddN@gradients_1/extrinsic_value/MatMul_grad/tuple/control_dependency6gradients_1/dense/MatMul_grad/tuple/control_dependency*
T0*A
_class7
53loc:@gradients_1/extrinsic_value/MatMul_grad/MatMul*
N
X
 gradients_1/Reshape_2_grad/ShapeShapelstm/rnn/transpose_1*
T0*
out_type0
z
"gradients_1/Reshape_2_grad/ReshapeReshapegradients_1/AddN_4 gradients_1/Reshape_2_grad/Shape*
T0*
Tshape0
h
7gradients_1/lstm/rnn/transpose_1_grad/InvertPermutationInvertPermutationlstm/rnn/concat_2*
T0
�
/gradients_1/lstm/rnn/transpose_1_grad/transpose	Transpose"gradients_1/Reshape_2_grad/Reshape7gradients_1/lstm/rnn/transpose_1_grad/InvertPermutation*
Tperm0*
T0
�
`gradients_1/lstm/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3lstm/rnn/TensorArraylstm/rnn/while/Exit_2*'
_class
loc:@lstm/rnn/TensorArray*
sourcegradients_1
�
\gradients_1/lstm/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/gradient_flowIdentitylstm/rnn/while/Exit_2a^gradients_1/lstm/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*'
_class
loc:@lstm/rnn/TensorArray
�
fgradients_1/lstm/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3`gradients_1/lstm/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3lstm/rnn/TensorArrayStack/range/gradients_1/lstm/rnn/transpose_1_grad/transpose\gradients_1/lstm/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/gradient_flow*
T0
E
gradients_1/zeros_like_5	ZerosLikelstm/rnn/while/Exit_3*
T0
E
gradients_1/zeros_like_6	ZerosLikelstm/rnn/while/Exit_4*
T0
�
-gradients_1/lstm/rnn/while/Exit_2_grad/b_exitEnterfgradients_1/lstm/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayScatter/TensorArrayScatterV3*
T0*
is_constant( *
parallel_iterations *8

frame_name*(gradients_1/lstm/rnn/while/while_context
�
-gradients_1/lstm/rnn/while/Exit_3_grad/b_exitEntergradients_1/zeros_like_5*
T0*
is_constant( *
parallel_iterations *8

frame_name*(gradients_1/lstm/rnn/while/while_context
�
-gradients_1/lstm/rnn/while/Exit_4_grad/b_exitEntergradients_1/zeros_like_6*
T0*
is_constant( *
parallel_iterations *8

frame_name*(gradients_1/lstm/rnn/while/while_context
�
1gradients_1/lstm/rnn/while/Switch_2_grad/b_switchMerge-gradients_1/lstm/rnn/while/Exit_2_grad/b_exit8gradients_1/lstm/rnn/while/Switch_2_grad_1/NextIteration*
T0*
N
�
1gradients_1/lstm/rnn/while/Switch_3_grad/b_switchMerge-gradients_1/lstm/rnn/while/Exit_3_grad/b_exit8gradients_1/lstm/rnn/while/Switch_3_grad_1/NextIteration*
T0*
N
�
1gradients_1/lstm/rnn/while/Switch_4_grad/b_switchMerge-gradients_1/lstm/rnn/while/Exit_4_grad/b_exit8gradients_1/lstm/rnn/while/Switch_4_grad_1/NextIteration*
T0*
N
�
.gradients_1/lstm/rnn/while/Merge_2_grad/SwitchSwitch1gradients_1/lstm/rnn/while/Switch_2_grad/b_switchgradients_1/b_count_2*
T0*D
_class:
86loc:@gradients_1/lstm/rnn/while/Switch_2_grad/b_switch
q
8gradients_1/lstm/rnn/while/Merge_2_grad/tuple/group_depsNoOp/^gradients_1/lstm/rnn/while/Merge_2_grad/Switch
�
@gradients_1/lstm/rnn/while/Merge_2_grad/tuple/control_dependencyIdentity.gradients_1/lstm/rnn/while/Merge_2_grad/Switch9^gradients_1/lstm/rnn/while/Merge_2_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients_1/lstm/rnn/while/Switch_2_grad/b_switch
�
Bgradients_1/lstm/rnn/while/Merge_2_grad/tuple/control_dependency_1Identity0gradients_1/lstm/rnn/while/Merge_2_grad/Switch:19^gradients_1/lstm/rnn/while/Merge_2_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients_1/lstm/rnn/while/Switch_2_grad/b_switch
�
.gradients_1/lstm/rnn/while/Merge_3_grad/SwitchSwitch1gradients_1/lstm/rnn/while/Switch_3_grad/b_switchgradients_1/b_count_2*
T0*D
_class:
86loc:@gradients_1/lstm/rnn/while/Switch_3_grad/b_switch
q
8gradients_1/lstm/rnn/while/Merge_3_grad/tuple/group_depsNoOp/^gradients_1/lstm/rnn/while/Merge_3_grad/Switch
�
@gradients_1/lstm/rnn/while/Merge_3_grad/tuple/control_dependencyIdentity.gradients_1/lstm/rnn/while/Merge_3_grad/Switch9^gradients_1/lstm/rnn/while/Merge_3_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients_1/lstm/rnn/while/Switch_3_grad/b_switch
�
Bgradients_1/lstm/rnn/while/Merge_3_grad/tuple/control_dependency_1Identity0gradients_1/lstm/rnn/while/Merge_3_grad/Switch:19^gradients_1/lstm/rnn/while/Merge_3_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients_1/lstm/rnn/while/Switch_3_grad/b_switch
�
.gradients_1/lstm/rnn/while/Merge_4_grad/SwitchSwitch1gradients_1/lstm/rnn/while/Switch_4_grad/b_switchgradients_1/b_count_2*
T0*D
_class:
86loc:@gradients_1/lstm/rnn/while/Switch_4_grad/b_switch
q
8gradients_1/lstm/rnn/while/Merge_4_grad/tuple/group_depsNoOp/^gradients_1/lstm/rnn/while/Merge_4_grad/Switch
�
@gradients_1/lstm/rnn/while/Merge_4_grad/tuple/control_dependencyIdentity.gradients_1/lstm/rnn/while/Merge_4_grad/Switch9^gradients_1/lstm/rnn/while/Merge_4_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients_1/lstm/rnn/while/Switch_4_grad/b_switch
�
Bgradients_1/lstm/rnn/while/Merge_4_grad/tuple/control_dependency_1Identity0gradients_1/lstm/rnn/while/Merge_4_grad/Switch:19^gradients_1/lstm/rnn/while/Merge_4_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients_1/lstm/rnn/while/Switch_4_grad/b_switch

,gradients_1/lstm/rnn/while/Enter_2_grad/ExitExit@gradients_1/lstm/rnn/while/Merge_2_grad/tuple/control_dependency*
T0

,gradients_1/lstm/rnn/while/Enter_3_grad/ExitExit@gradients_1/lstm/rnn/while/Merge_3_grad/tuple/control_dependency*
T0

,gradients_1/lstm/rnn/while/Enter_4_grad/ExitExit@gradients_1/lstm/rnn/while/Merge_4_grad/tuple/control_dependency*
T0
�
egradients_1/lstm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3kgradients_1/lstm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterBgradients_1/lstm/rnn/while/Merge_2_grad/tuple/control_dependency_1*7
_class-
+)loc:@lstm/rnn/while/basic_lstm_cell/Mul_2*
sourcegradients_1
�
kgradients_1/lstm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterEnterlstm/rnn/TensorArray*
T0*7
_class-
+)loc:@lstm/rnn/while/basic_lstm_cell/Mul_2*
is_constant(*
parallel_iterations *8

frame_name*(gradients_1/lstm/rnn/while/while_context
�
agradients_1/lstm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/gradient_flowIdentityBgradients_1/lstm/rnn/while/Merge_2_grad/tuple/control_dependency_1f^gradients_1/lstm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*7
_class-
+)loc:@lstm/rnn/while/basic_lstm_cell/Mul_2
�
Ugradients_1/lstm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3TensorArrayReadV3egradients_1/lstm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3`gradients_1/lstm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2agradients_1/lstm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/gradient_flow*
dtype0
�
[gradients_1/lstm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/ConstConst*,
_class"
 loc:@lstm/rnn/while/Identity_1*
valueB :
���������*
dtype0
�
[gradients_1/lstm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_accStackV2[gradients_1/lstm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Const*
	elem_type0*,
_class"
 loc:@lstm/rnn/while/Identity_1*

stack_name 
�
[gradients_1/lstm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/EnterEnter[gradients_1/lstm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc*
T0*
is_constant(*
parallel_iterations *,

frame_namelstm/rnn/while/while_context
�
agradients_1/lstm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2StackPushV2[gradients_1/lstm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Enterlstm/rnn/while/Identity_1^gradients_1/Add*
T0*
swap_memory( 
�
`gradients_1/lstm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2
StackPopV2fgradients_1/lstm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2/Enter^gradients_1/Sub*
	elem_type0
�
fgradients_1/lstm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2/EnterEnter[gradients_1/lstm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc*
T0*
is_constant(*
parallel_iterations *8

frame_name*(gradients_1/lstm/rnn/while/while_context
�
\gradients_1/lstm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/b_syncControlTriggera^gradients_1/lstm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2W^gradients_1/lstm/rnn/while/basic_lstm_cell/Add_1_grad/BroadcastGradientArgs/StackPopV2Y^gradients_1/lstm/rnn/while/basic_lstm_cell/Add_1_grad/BroadcastGradientArgs/StackPopV2_1U^gradients_1/lstm/rnn/while/basic_lstm_cell/Add_grad/BroadcastGradientArgs/StackPopV2W^gradients_1/lstm/rnn/while/basic_lstm_cell/Add_grad/BroadcastGradientArgs/StackPopV2_1K^gradients_1/lstm/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2W^gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/BroadcastGradientArgs/StackPopV2Y^gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/BroadcastGradientArgs/StackPopV2_1E^gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/StackPopV2G^gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/StackPopV2W^gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/BroadcastGradientArgs/StackPopV2Y^gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/BroadcastGradientArgs/StackPopV2_1E^gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/StackPopV2G^gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/StackPopV2U^gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_grad/BroadcastGradientArgs/StackPopV2W^gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_grad/BroadcastGradientArgs/StackPopV2_1C^gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_grad/Mul/StackPopV2E^gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/StackPopV2I^gradients_1/lstm/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPopV2K^gradients_1/lstm/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPopV2_1
�
Tgradients_1/lstm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/group_depsNoOpC^gradients_1/lstm/rnn/while/Merge_2_grad/tuple/control_dependency_1V^gradients_1/lstm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3
�
\gradients_1/lstm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependencyIdentityUgradients_1/lstm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3U^gradients_1/lstm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/group_deps*
T0*h
_class^
\Zloc:@gradients_1/lstm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3
�
^gradients_1/lstm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency_1IdentityBgradients_1/lstm/rnn/while/Merge_2_grad/tuple/control_dependency_1U^gradients_1/lstm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients_1/lstm/rnn/while/Switch_2_grad/b_switch
�
gradients_1/AddN_5AddNBgradients_1/lstm/rnn/while/Merge_4_grad/tuple/control_dependency_1\gradients_1/lstm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency*
T0*D
_class:
86loc:@gradients_1/lstm/rnn/while/Switch_4_grad/b_switch*
N
�
;gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/ShapeShape%lstm/rnn/while/basic_lstm_cell/Tanh_1*
T0*
out_type0
�
=gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/Shape_1Shape(lstm/rnn/while/basic_lstm_cell/Sigmoid_2*
T0*
out_type0
�
Kgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsVgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/BroadcastGradientArgs/StackPopV2Xgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/BroadcastGradientArgs/StackPopV2_1*
T0
�
Qgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/BroadcastGradientArgs/ConstConst*N
_classD
B@loc:@gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/Shape*
valueB :
���������*
dtype0
�
Qgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/BroadcastGradientArgs/f_accStackV2Qgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/BroadcastGradientArgs/Const*
	elem_type0*N
_classD
B@loc:@gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/Shape*

stack_name 
�
Qgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/BroadcastGradientArgs/EnterEnterQgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *,

frame_namelstm/rnn/while/while_context
�
Wgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/BroadcastGradientArgs/StackPushV2StackPushV2Qgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/BroadcastGradientArgs/Enter;gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/Shape^gradients_1/Add*
T0*
swap_memory( 
�
Vgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/BroadcastGradientArgs/StackPopV2
StackPopV2\gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients_1/Sub*
	elem_type0
�
\gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/BroadcastGradientArgs/StackPopV2/EnterEnterQgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *8

frame_name*(gradients_1/lstm/rnn/while/while_context
�
Sgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/BroadcastGradientArgs/Const_1Const*P
_classF
DBloc:@gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/Shape_1*
valueB :
���������*
dtype0
�
Sgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/BroadcastGradientArgs/f_acc_1StackV2Sgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/BroadcastGradientArgs/Const_1*
	elem_type0*P
_classF
DBloc:@gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/Shape_1*

stack_name 
�
Sgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/BroadcastGradientArgs/Enter_1EnterSgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *,

frame_namelstm/rnn/while/while_context
�
Ygradients_1/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2Sgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/BroadcastGradientArgs/Enter_1=gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/Shape_1^gradients_1/Add*
T0*
swap_memory( 
�
Xgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2^gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/BroadcastGradientArgs/StackPopV2_1/Enter^gradients_1/Sub*
	elem_type0
�
^gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterSgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *8

frame_name*(gradients_1/lstm/rnn/while/while_context
�
9gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/MulMulgradients_1/AddN_5Dgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/StackPopV2*
T0
�
?gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/ConstConst*;
_class1
/-loc:@lstm/rnn/while/basic_lstm_cell/Sigmoid_2*
valueB :
���������*
dtype0
�
?gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/f_accStackV2?gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/Const*
	elem_type0*;
_class1
/-loc:@lstm/rnn/while/basic_lstm_cell/Sigmoid_2*

stack_name 
�
?gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/EnterEnter?gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *,

frame_namelstm/rnn/while/while_context
�
Egradients_1/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/StackPushV2StackPushV2?gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/Enter(lstm/rnn/while/basic_lstm_cell/Sigmoid_2^gradients_1/Add*
T0*
swap_memory( 
�
Dgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/StackPopV2
StackPopV2Jgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/StackPopV2/Enter^gradients_1/Sub*
	elem_type0
�
Jgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/StackPopV2/EnterEnter?gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *8

frame_name*(gradients_1/lstm/rnn/while/while_context
�
9gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/SumSum9gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/MulKgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
�
=gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/ReshapeReshape9gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/SumVgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0
�
;gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1MulFgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/StackPopV2gradients_1/AddN_5*
T0
�
Agradients_1/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/ConstConst*8
_class.
,*loc:@lstm/rnn/while/basic_lstm_cell/Tanh_1*
valueB :
���������*
dtype0
�
Agradients_1/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/f_accStackV2Agradients_1/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/Const*
	elem_type0*8
_class.
,*loc:@lstm/rnn/while/basic_lstm_cell/Tanh_1*

stack_name 
�
Agradients_1/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/EnterEnterAgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *,

frame_namelstm/rnn/while/while_context
�
Ggradients_1/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/StackPushV2StackPushV2Agradients_1/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/Enter%lstm/rnn/while/basic_lstm_cell/Tanh_1^gradients_1/Add*
T0*
swap_memory( 
�
Fgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/StackPopV2
StackPopV2Lgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/StackPopV2/Enter^gradients_1/Sub*
	elem_type0
�
Lgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/StackPopV2/EnterEnterAgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *8

frame_name*(gradients_1/lstm/rnn/while/while_context
�
;gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/Sum_1Sum;gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1Mgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
�
?gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/Reshape_1Reshape;gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/Sum_1Xgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/BroadcastGradientArgs/StackPopV2_1*
T0*
Tshape0
�
Fgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/tuple/group_depsNoOp>^gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/Reshape@^gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/Reshape_1
�
Ngradients_1/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/tuple/control_dependencyIdentity=gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/ReshapeG^gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/tuple/group_deps*
T0*P
_classF
DBloc:@gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/Reshape
�
Pgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/tuple/control_dependency_1Identity?gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/Reshape_1G^gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/tuple/group_deps*
T0*R
_classH
FDloc:@gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/Reshape_1
�
?gradients_1/lstm/rnn/while/basic_lstm_cell/Tanh_1_grad/TanhGradTanhGradFgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/Mul_1/StackPopV2Ngradients_1/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/tuple/control_dependency*
T0
�
Egradients_1/lstm/rnn/while/basic_lstm_cell/Sigmoid_2_grad/SigmoidGradSigmoidGradDgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/Mul/StackPopV2Pgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_2_grad/tuple/control_dependency_1*
T0
�
8gradients_1/lstm/rnn/while/Switch_2_grad_1/NextIterationNextIteration^gradients_1/lstm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency_1*
T0
�
gradients_1/AddN_6AddNBgradients_1/lstm/rnn/while/Merge_3_grad/tuple/control_dependency_1?gradients_1/lstm/rnn/while/basic_lstm_cell/Tanh_1_grad/TanhGrad*
T0*D
_class:
86loc:@gradients_1/lstm/rnn/while/Switch_3_grad/b_switch*
N
�
;gradients_1/lstm/rnn/while/basic_lstm_cell/Add_1_grad/ShapeShape"lstm/rnn/while/basic_lstm_cell/Mul*
T0*
out_type0
�
=gradients_1/lstm/rnn/while/basic_lstm_cell/Add_1_grad/Shape_1Shape$lstm/rnn/while/basic_lstm_cell/Mul_1*
T0*
out_type0
�
Kgradients_1/lstm/rnn/while/basic_lstm_cell/Add_1_grad/BroadcastGradientArgsBroadcastGradientArgsVgradients_1/lstm/rnn/while/basic_lstm_cell/Add_1_grad/BroadcastGradientArgs/StackPopV2Xgradients_1/lstm/rnn/while/basic_lstm_cell/Add_1_grad/BroadcastGradientArgs/StackPopV2_1*
T0
�
Qgradients_1/lstm/rnn/while/basic_lstm_cell/Add_1_grad/BroadcastGradientArgs/ConstConst*N
_classD
B@loc:@gradients_1/lstm/rnn/while/basic_lstm_cell/Add_1_grad/Shape*
valueB :
���������*
dtype0
�
Qgradients_1/lstm/rnn/while/basic_lstm_cell/Add_1_grad/BroadcastGradientArgs/f_accStackV2Qgradients_1/lstm/rnn/while/basic_lstm_cell/Add_1_grad/BroadcastGradientArgs/Const*
	elem_type0*N
_classD
B@loc:@gradients_1/lstm/rnn/while/basic_lstm_cell/Add_1_grad/Shape*

stack_name 
�
Qgradients_1/lstm/rnn/while/basic_lstm_cell/Add_1_grad/BroadcastGradientArgs/EnterEnterQgradients_1/lstm/rnn/while/basic_lstm_cell/Add_1_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *,

frame_namelstm/rnn/while/while_context
�
Wgradients_1/lstm/rnn/while/basic_lstm_cell/Add_1_grad/BroadcastGradientArgs/StackPushV2StackPushV2Qgradients_1/lstm/rnn/while/basic_lstm_cell/Add_1_grad/BroadcastGradientArgs/Enter;gradients_1/lstm/rnn/while/basic_lstm_cell/Add_1_grad/Shape^gradients_1/Add*
T0*
swap_memory( 
�
Vgradients_1/lstm/rnn/while/basic_lstm_cell/Add_1_grad/BroadcastGradientArgs/StackPopV2
StackPopV2\gradients_1/lstm/rnn/while/basic_lstm_cell/Add_1_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients_1/Sub*
	elem_type0
�
\gradients_1/lstm/rnn/while/basic_lstm_cell/Add_1_grad/BroadcastGradientArgs/StackPopV2/EnterEnterQgradients_1/lstm/rnn/while/basic_lstm_cell/Add_1_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *8

frame_name*(gradients_1/lstm/rnn/while/while_context
�
Sgradients_1/lstm/rnn/while/basic_lstm_cell/Add_1_grad/BroadcastGradientArgs/Const_1Const*P
_classF
DBloc:@gradients_1/lstm/rnn/while/basic_lstm_cell/Add_1_grad/Shape_1*
valueB :
���������*
dtype0
�
Sgradients_1/lstm/rnn/while/basic_lstm_cell/Add_1_grad/BroadcastGradientArgs/f_acc_1StackV2Sgradients_1/lstm/rnn/while/basic_lstm_cell/Add_1_grad/BroadcastGradientArgs/Const_1*
	elem_type0*P
_classF
DBloc:@gradients_1/lstm/rnn/while/basic_lstm_cell/Add_1_grad/Shape_1*

stack_name 
�
Sgradients_1/lstm/rnn/while/basic_lstm_cell/Add_1_grad/BroadcastGradientArgs/Enter_1EnterSgradients_1/lstm/rnn/while/basic_lstm_cell/Add_1_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *,

frame_namelstm/rnn/while/while_context
�
Ygradients_1/lstm/rnn/while/basic_lstm_cell/Add_1_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2Sgradients_1/lstm/rnn/while/basic_lstm_cell/Add_1_grad/BroadcastGradientArgs/Enter_1=gradients_1/lstm/rnn/while/basic_lstm_cell/Add_1_grad/Shape_1^gradients_1/Add*
T0*
swap_memory( 
�
Xgradients_1/lstm/rnn/while/basic_lstm_cell/Add_1_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2^gradients_1/lstm/rnn/while/basic_lstm_cell/Add_1_grad/BroadcastGradientArgs/StackPopV2_1/Enter^gradients_1/Sub*
	elem_type0
�
^gradients_1/lstm/rnn/while/basic_lstm_cell/Add_1_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterSgradients_1/lstm/rnn/while/basic_lstm_cell/Add_1_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *8

frame_name*(gradients_1/lstm/rnn/while/while_context
�
9gradients_1/lstm/rnn/while/basic_lstm_cell/Add_1_grad/SumSumgradients_1/AddN_6Kgradients_1/lstm/rnn/while/basic_lstm_cell/Add_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
�
=gradients_1/lstm/rnn/while/basic_lstm_cell/Add_1_grad/ReshapeReshape9gradients_1/lstm/rnn/while/basic_lstm_cell/Add_1_grad/SumVgradients_1/lstm/rnn/while/basic_lstm_cell/Add_1_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0
�
;gradients_1/lstm/rnn/while/basic_lstm_cell/Add_1_grad/Sum_1Sumgradients_1/AddN_6Mgradients_1/lstm/rnn/while/basic_lstm_cell/Add_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
�
?gradients_1/lstm/rnn/while/basic_lstm_cell/Add_1_grad/Reshape_1Reshape;gradients_1/lstm/rnn/while/basic_lstm_cell/Add_1_grad/Sum_1Xgradients_1/lstm/rnn/while/basic_lstm_cell/Add_1_grad/BroadcastGradientArgs/StackPopV2_1*
T0*
Tshape0
�
Fgradients_1/lstm/rnn/while/basic_lstm_cell/Add_1_grad/tuple/group_depsNoOp>^gradients_1/lstm/rnn/while/basic_lstm_cell/Add_1_grad/Reshape@^gradients_1/lstm/rnn/while/basic_lstm_cell/Add_1_grad/Reshape_1
�
Ngradients_1/lstm/rnn/while/basic_lstm_cell/Add_1_grad/tuple/control_dependencyIdentity=gradients_1/lstm/rnn/while/basic_lstm_cell/Add_1_grad/ReshapeG^gradients_1/lstm/rnn/while/basic_lstm_cell/Add_1_grad/tuple/group_deps*
T0*P
_classF
DBloc:@gradients_1/lstm/rnn/while/basic_lstm_cell/Add_1_grad/Reshape
�
Pgradients_1/lstm/rnn/while/basic_lstm_cell/Add_1_grad/tuple/control_dependency_1Identity?gradients_1/lstm/rnn/while/basic_lstm_cell/Add_1_grad/Reshape_1G^gradients_1/lstm/rnn/while/basic_lstm_cell/Add_1_grad/tuple/group_deps*
T0*R
_classH
FDloc:@gradients_1/lstm/rnn/while/basic_lstm_cell/Add_1_grad/Reshape_1
v
9gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_grad/ShapeShapelstm/rnn/while/Identity_3*
T0*
out_type0
�
;gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_grad/Shape_1Shape&lstm/rnn/while/basic_lstm_cell/Sigmoid*
T0*
out_type0
�
Igradients_1/lstm/rnn/while/basic_lstm_cell/Mul_grad/BroadcastGradientArgsBroadcastGradientArgsTgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_grad/BroadcastGradientArgs/StackPopV2Vgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_grad/BroadcastGradientArgs/StackPopV2_1*
T0
�
Ogradients_1/lstm/rnn/while/basic_lstm_cell/Mul_grad/BroadcastGradientArgs/ConstConst*L
_classB
@>loc:@gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_grad/Shape*
valueB :
���������*
dtype0
�
Ogradients_1/lstm/rnn/while/basic_lstm_cell/Mul_grad/BroadcastGradientArgs/f_accStackV2Ogradients_1/lstm/rnn/while/basic_lstm_cell/Mul_grad/BroadcastGradientArgs/Const*
	elem_type0*L
_classB
@>loc:@gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_grad/Shape*

stack_name 
�
Ogradients_1/lstm/rnn/while/basic_lstm_cell/Mul_grad/BroadcastGradientArgs/EnterEnterOgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *,

frame_namelstm/rnn/while/while_context
�
Ugradients_1/lstm/rnn/while/basic_lstm_cell/Mul_grad/BroadcastGradientArgs/StackPushV2StackPushV2Ogradients_1/lstm/rnn/while/basic_lstm_cell/Mul_grad/BroadcastGradientArgs/Enter9gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_grad/Shape^gradients_1/Add*
T0*
swap_memory( 
�
Tgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_grad/BroadcastGradientArgs/StackPopV2
StackPopV2Zgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients_1/Sub*
	elem_type0
�
Zgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_grad/BroadcastGradientArgs/StackPopV2/EnterEnterOgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *8

frame_name*(gradients_1/lstm/rnn/while/while_context
�
Qgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_grad/BroadcastGradientArgs/Const_1Const*N
_classD
B@loc:@gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_grad/Shape_1*
valueB :
���������*
dtype0
�
Qgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_grad/BroadcastGradientArgs/f_acc_1StackV2Qgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_grad/BroadcastGradientArgs/Const_1*
	elem_type0*N
_classD
B@loc:@gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_grad/Shape_1*

stack_name 
�
Qgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_grad/BroadcastGradientArgs/Enter_1EnterQgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *,

frame_namelstm/rnn/while/while_context
�
Wgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2Qgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_grad/BroadcastGradientArgs/Enter_1;gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_grad/Shape_1^gradients_1/Add*
T0*
swap_memory( 
�
Vgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2\gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_grad/BroadcastGradientArgs/StackPopV2_1/Enter^gradients_1/Sub*
	elem_type0
�
\gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterQgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *8

frame_name*(gradients_1/lstm/rnn/while/while_context
�
7gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_grad/MulMulNgradients_1/lstm/rnn/while/basic_lstm_cell/Add_1_grad/tuple/control_dependencyBgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_grad/Mul/StackPopV2*
T0
�
=gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_grad/Mul/ConstConst*9
_class/
-+loc:@lstm/rnn/while/basic_lstm_cell/Sigmoid*
valueB :
���������*
dtype0
�
=gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_grad/Mul/f_accStackV2=gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_grad/Mul/Const*
	elem_type0*9
_class/
-+loc:@lstm/rnn/while/basic_lstm_cell/Sigmoid*

stack_name 
�
=gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_grad/Mul/EnterEnter=gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *,

frame_namelstm/rnn/while/while_context
�
Cgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_grad/Mul/StackPushV2StackPushV2=gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_grad/Mul/Enter&lstm/rnn/while/basic_lstm_cell/Sigmoid^gradients_1/Add*
T0*
swap_memory( 
�
Bgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_grad/Mul/StackPopV2
StackPopV2Hgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_grad/Mul/StackPopV2/Enter^gradients_1/Sub*
	elem_type0
�
Hgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_grad/Mul/StackPopV2/EnterEnter=gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *8

frame_name*(gradients_1/lstm/rnn/while/while_context
�
7gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_grad/SumSum7gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_grad/MulIgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
�
;gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_grad/ReshapeReshape7gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_grad/SumTgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0
�
9gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_grad/Mul_1MulDgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/StackPopV2Ngradients_1/lstm/rnn/while/basic_lstm_cell/Add_1_grad/tuple/control_dependency*
T0
�
?gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/ConstConst*,
_class"
 loc:@lstm/rnn/while/Identity_3*
valueB :
���������*
dtype0
�
?gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/f_accStackV2?gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/Const*
	elem_type0*,
_class"
 loc:@lstm/rnn/while/Identity_3*

stack_name 
�
?gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/EnterEnter?gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *,

frame_namelstm/rnn/while/while_context
�
Egradients_1/lstm/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/StackPushV2StackPushV2?gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/Enterlstm/rnn/while/Identity_3^gradients_1/Add*
T0*
swap_memory( 
�
Dgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/StackPopV2
StackPopV2Jgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/StackPopV2/Enter^gradients_1/Sub*
	elem_type0
�
Jgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/StackPopV2/EnterEnter?gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *8

frame_name*(gradients_1/lstm/rnn/while/while_context
�
9gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_grad/Sum_1Sum9gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_grad/Mul_1Kgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
�
=gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_grad/Reshape_1Reshape9gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_grad/Sum_1Vgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_grad/BroadcastGradientArgs/StackPopV2_1*
T0*
Tshape0
�
Dgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_grad/tuple/group_depsNoOp<^gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_grad/Reshape>^gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_grad/Reshape_1
�
Lgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_grad/tuple/control_dependencyIdentity;gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_grad/ReshapeE^gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_grad/tuple/group_deps*
T0*N
_classD
B@loc:@gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_grad/Reshape
�
Ngradients_1/lstm/rnn/while/basic_lstm_cell/Mul_grad/tuple/control_dependency_1Identity=gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_grad/Reshape_1E^gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_grad/tuple/group_deps*
T0*P
_classF
DBloc:@gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_grad/Reshape_1
�
;gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/ShapeShape(lstm/rnn/while/basic_lstm_cell/Sigmoid_1*
T0*
out_type0
�
=gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/Shape_1Shape#lstm/rnn/while/basic_lstm_cell/Tanh*
T0*
out_type0
�
Kgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsVgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/BroadcastGradientArgs/StackPopV2Xgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/BroadcastGradientArgs/StackPopV2_1*
T0
�
Qgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/BroadcastGradientArgs/ConstConst*N
_classD
B@loc:@gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/Shape*
valueB :
���������*
dtype0
�
Qgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/BroadcastGradientArgs/f_accStackV2Qgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/BroadcastGradientArgs/Const*
	elem_type0*N
_classD
B@loc:@gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/Shape*

stack_name 
�
Qgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/BroadcastGradientArgs/EnterEnterQgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *,

frame_namelstm/rnn/while/while_context
�
Wgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/BroadcastGradientArgs/StackPushV2StackPushV2Qgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/BroadcastGradientArgs/Enter;gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/Shape^gradients_1/Add*
T0*
swap_memory( 
�
Vgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/BroadcastGradientArgs/StackPopV2
StackPopV2\gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients_1/Sub*
	elem_type0
�
\gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/BroadcastGradientArgs/StackPopV2/EnterEnterQgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *8

frame_name*(gradients_1/lstm/rnn/while/while_context
�
Sgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/BroadcastGradientArgs/Const_1Const*P
_classF
DBloc:@gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/Shape_1*
valueB :
���������*
dtype0
�
Sgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/BroadcastGradientArgs/f_acc_1StackV2Sgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/BroadcastGradientArgs/Const_1*
	elem_type0*P
_classF
DBloc:@gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/Shape_1*

stack_name 
�
Sgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/BroadcastGradientArgs/Enter_1EnterSgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *,

frame_namelstm/rnn/while/while_context
�
Ygradients_1/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2Sgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/BroadcastGradientArgs/Enter_1=gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/Shape_1^gradients_1/Add*
T0*
swap_memory( 
�
Xgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2^gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/BroadcastGradientArgs/StackPopV2_1/Enter^gradients_1/Sub*
	elem_type0
�
^gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterSgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *8

frame_name*(gradients_1/lstm/rnn/while/while_context
�
9gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/MulMulPgradients_1/lstm/rnn/while/basic_lstm_cell/Add_1_grad/tuple/control_dependency_1Dgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/StackPopV2*
T0
�
?gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/ConstConst*6
_class,
*(loc:@lstm/rnn/while/basic_lstm_cell/Tanh*
valueB :
���������*
dtype0
�
?gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/f_accStackV2?gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/Const*
	elem_type0*6
_class,
*(loc:@lstm/rnn/while/basic_lstm_cell/Tanh*

stack_name 
�
?gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/EnterEnter?gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *,

frame_namelstm/rnn/while/while_context
�
Egradients_1/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/StackPushV2StackPushV2?gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/Enter#lstm/rnn/while/basic_lstm_cell/Tanh^gradients_1/Add*
T0*
swap_memory( 
�
Dgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/StackPopV2
StackPopV2Jgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/StackPopV2/Enter^gradients_1/Sub*
	elem_type0
�
Jgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/StackPopV2/EnterEnter?gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/f_acc*
T0*
is_constant(*
parallel_iterations *8

frame_name*(gradients_1/lstm/rnn/while/while_context
�
9gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/SumSum9gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/MulKgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
�
=gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/ReshapeReshape9gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/SumVgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0
�
;gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1MulFgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/StackPopV2Pgradients_1/lstm/rnn/while/basic_lstm_cell/Add_1_grad/tuple/control_dependency_1*
T0
�
Agradients_1/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/ConstConst*;
_class1
/-loc:@lstm/rnn/while/basic_lstm_cell/Sigmoid_1*
valueB :
���������*
dtype0
�
Agradients_1/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/f_accStackV2Agradients_1/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/Const*
	elem_type0*;
_class1
/-loc:@lstm/rnn/while/basic_lstm_cell/Sigmoid_1*

stack_name 
�
Agradients_1/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/EnterEnterAgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *,

frame_namelstm/rnn/while/while_context
�
Ggradients_1/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/StackPushV2StackPushV2Agradients_1/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/Enter(lstm/rnn/while/basic_lstm_cell/Sigmoid_1^gradients_1/Add*
T0*
swap_memory( 
�
Fgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/StackPopV2
StackPopV2Lgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/StackPopV2/Enter^gradients_1/Sub*
	elem_type0
�
Lgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/StackPopV2/EnterEnterAgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/f_acc*
T0*
is_constant(*
parallel_iterations *8

frame_name*(gradients_1/lstm/rnn/while/while_context
�
;gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/Sum_1Sum;gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1Mgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
�
?gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/Reshape_1Reshape;gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/Sum_1Xgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/BroadcastGradientArgs/StackPopV2_1*
T0*
Tshape0
�
Fgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/tuple/group_depsNoOp>^gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/Reshape@^gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/Reshape_1
�
Ngradients_1/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/tuple/control_dependencyIdentity=gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/ReshapeG^gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/tuple/group_deps*
T0*P
_classF
DBloc:@gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/Reshape
�
Pgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/tuple/control_dependency_1Identity?gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/Reshape_1G^gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/tuple/group_deps*
T0*R
_classH
FDloc:@gradients_1/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/Reshape_1
�
Cgradients_1/lstm/rnn/while/basic_lstm_cell/Sigmoid_grad/SigmoidGradSigmoidGradBgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_grad/Mul/StackPopV2Ngradients_1/lstm/rnn/while/basic_lstm_cell/Mul_grad/tuple/control_dependency_1*
T0
�
Egradients_1/lstm/rnn/while/basic_lstm_cell/Sigmoid_1_grad/SigmoidGradSigmoidGradFgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/Mul_1/StackPopV2Ngradients_1/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/tuple/control_dependency*
T0
�
=gradients_1/lstm/rnn/while/basic_lstm_cell/Tanh_grad/TanhGradTanhGradDgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/Mul/StackPopV2Pgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_1_grad/tuple/control_dependency_1*
T0
�
8gradients_1/lstm/rnn/while/Switch_3_grad_1/NextIterationNextIterationLgradients_1/lstm/rnn/while/basic_lstm_cell/Mul_grad/tuple/control_dependency*
T0
�
9gradients_1/lstm/rnn/while/basic_lstm_cell/Add_grad/ShapeShape&lstm/rnn/while/basic_lstm_cell/split:2*
T0*
out_type0
�
;gradients_1/lstm/rnn/while/basic_lstm_cell/Add_grad/Shape_1Shape&lstm/rnn/while/basic_lstm_cell/Const_2*
T0*
out_type0
�
Igradients_1/lstm/rnn/while/basic_lstm_cell/Add_grad/BroadcastGradientArgsBroadcastGradientArgsTgradients_1/lstm/rnn/while/basic_lstm_cell/Add_grad/BroadcastGradientArgs/StackPopV2Vgradients_1/lstm/rnn/while/basic_lstm_cell/Add_grad/BroadcastGradientArgs/StackPopV2_1*
T0
�
Ogradients_1/lstm/rnn/while/basic_lstm_cell/Add_grad/BroadcastGradientArgs/ConstConst*L
_classB
@>loc:@gradients_1/lstm/rnn/while/basic_lstm_cell/Add_grad/Shape*
valueB :
���������*
dtype0
�
Ogradients_1/lstm/rnn/while/basic_lstm_cell/Add_grad/BroadcastGradientArgs/f_accStackV2Ogradients_1/lstm/rnn/while/basic_lstm_cell/Add_grad/BroadcastGradientArgs/Const*
	elem_type0*L
_classB
@>loc:@gradients_1/lstm/rnn/while/basic_lstm_cell/Add_grad/Shape*

stack_name 
�
Ogradients_1/lstm/rnn/while/basic_lstm_cell/Add_grad/BroadcastGradientArgs/EnterEnterOgradients_1/lstm/rnn/while/basic_lstm_cell/Add_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *,

frame_namelstm/rnn/while/while_context
�
Ugradients_1/lstm/rnn/while/basic_lstm_cell/Add_grad/BroadcastGradientArgs/StackPushV2StackPushV2Ogradients_1/lstm/rnn/while/basic_lstm_cell/Add_grad/BroadcastGradientArgs/Enter9gradients_1/lstm/rnn/while/basic_lstm_cell/Add_grad/Shape^gradients_1/Add*
T0*
swap_memory( 
�
Tgradients_1/lstm/rnn/while/basic_lstm_cell/Add_grad/BroadcastGradientArgs/StackPopV2
StackPopV2Zgradients_1/lstm/rnn/while/basic_lstm_cell/Add_grad/BroadcastGradientArgs/StackPopV2/Enter^gradients_1/Sub*
	elem_type0
�
Zgradients_1/lstm/rnn/while/basic_lstm_cell/Add_grad/BroadcastGradientArgs/StackPopV2/EnterEnterOgradients_1/lstm/rnn/while/basic_lstm_cell/Add_grad/BroadcastGradientArgs/f_acc*
T0*
is_constant(*
parallel_iterations *8

frame_name*(gradients_1/lstm/rnn/while/while_context
�
Qgradients_1/lstm/rnn/while/basic_lstm_cell/Add_grad/BroadcastGradientArgs/Const_1Const*N
_classD
B@loc:@gradients_1/lstm/rnn/while/basic_lstm_cell/Add_grad/Shape_1*
valueB :
���������*
dtype0
�
Qgradients_1/lstm/rnn/while/basic_lstm_cell/Add_grad/BroadcastGradientArgs/f_acc_1StackV2Qgradients_1/lstm/rnn/while/basic_lstm_cell/Add_grad/BroadcastGradientArgs/Const_1*
	elem_type0*N
_classD
B@loc:@gradients_1/lstm/rnn/while/basic_lstm_cell/Add_grad/Shape_1*

stack_name 
�
Qgradients_1/lstm/rnn/while/basic_lstm_cell/Add_grad/BroadcastGradientArgs/Enter_1EnterQgradients_1/lstm/rnn/while/basic_lstm_cell/Add_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *,

frame_namelstm/rnn/while/while_context
�
Wgradients_1/lstm/rnn/while/basic_lstm_cell/Add_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2Qgradients_1/lstm/rnn/while/basic_lstm_cell/Add_grad/BroadcastGradientArgs/Enter_1;gradients_1/lstm/rnn/while/basic_lstm_cell/Add_grad/Shape_1^gradients_1/Add*
T0*
swap_memory( 
�
Vgradients_1/lstm/rnn/while/basic_lstm_cell/Add_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2\gradients_1/lstm/rnn/while/basic_lstm_cell/Add_grad/BroadcastGradientArgs/StackPopV2_1/Enter^gradients_1/Sub*
	elem_type0
�
\gradients_1/lstm/rnn/while/basic_lstm_cell/Add_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnterQgradients_1/lstm/rnn/while/basic_lstm_cell/Add_grad/BroadcastGradientArgs/f_acc_1*
T0*
is_constant(*
parallel_iterations *8

frame_name*(gradients_1/lstm/rnn/while/while_context
�
7gradients_1/lstm/rnn/while/basic_lstm_cell/Add_grad/SumSumCgradients_1/lstm/rnn/while/basic_lstm_cell/Sigmoid_grad/SigmoidGradIgradients_1/lstm/rnn/while/basic_lstm_cell/Add_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
�
;gradients_1/lstm/rnn/while/basic_lstm_cell/Add_grad/ReshapeReshape7gradients_1/lstm/rnn/while/basic_lstm_cell/Add_grad/SumTgradients_1/lstm/rnn/while/basic_lstm_cell/Add_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0
�
9gradients_1/lstm/rnn/while/basic_lstm_cell/Add_grad/Sum_1SumCgradients_1/lstm/rnn/while/basic_lstm_cell/Sigmoid_grad/SigmoidGradKgradients_1/lstm/rnn/while/basic_lstm_cell/Add_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
�
=gradients_1/lstm/rnn/while/basic_lstm_cell/Add_grad/Reshape_1Reshape9gradients_1/lstm/rnn/while/basic_lstm_cell/Add_grad/Sum_1Vgradients_1/lstm/rnn/while/basic_lstm_cell/Add_grad/BroadcastGradientArgs/StackPopV2_1*
T0*
Tshape0
�
Dgradients_1/lstm/rnn/while/basic_lstm_cell/Add_grad/tuple/group_depsNoOp<^gradients_1/lstm/rnn/while/basic_lstm_cell/Add_grad/Reshape>^gradients_1/lstm/rnn/while/basic_lstm_cell/Add_grad/Reshape_1
�
Lgradients_1/lstm/rnn/while/basic_lstm_cell/Add_grad/tuple/control_dependencyIdentity;gradients_1/lstm/rnn/while/basic_lstm_cell/Add_grad/ReshapeE^gradients_1/lstm/rnn/while/basic_lstm_cell/Add_grad/tuple/group_deps*
T0*N
_classD
B@loc:@gradients_1/lstm/rnn/while/basic_lstm_cell/Add_grad/Reshape
�
Ngradients_1/lstm/rnn/while/basic_lstm_cell/Add_grad/tuple/control_dependency_1Identity=gradients_1/lstm/rnn/while/basic_lstm_cell/Add_grad/Reshape_1E^gradients_1/lstm/rnn/while/basic_lstm_cell/Add_grad/tuple/group_deps*
T0*P
_classF
DBloc:@gradients_1/lstm/rnn/while/basic_lstm_cell/Add_grad/Reshape_1
�
<gradients_1/lstm/rnn/while/basic_lstm_cell/split_grad/concatConcatV2Egradients_1/lstm/rnn/while/basic_lstm_cell/Sigmoid_1_grad/SigmoidGrad=gradients_1/lstm/rnn/while/basic_lstm_cell/Tanh_grad/TanhGradLgradients_1/lstm/rnn/while/basic_lstm_cell/Add_grad/tuple/control_dependencyEgradients_1/lstm/rnn/while/basic_lstm_cell/Sigmoid_2_grad/SigmoidGradBgradients_1/lstm/rnn/while/basic_lstm_cell/split_grad/concat/Const*

Tidx0*
T0*
N
~
Bgradients_1/lstm/rnn/while/basic_lstm_cell/split_grad/concat/ConstConst^gradients_1/Sub*
value	B :*
dtype0
�
Cgradients_1/lstm/rnn/while/basic_lstm_cell/BiasAdd_grad/BiasAddGradBiasAddGrad<gradients_1/lstm/rnn/while/basic_lstm_cell/split_grad/concat*
T0*
data_formatNHWC
�
Hgradients_1/lstm/rnn/while/basic_lstm_cell/BiasAdd_grad/tuple/group_depsNoOpD^gradients_1/lstm/rnn/while/basic_lstm_cell/BiasAdd_grad/BiasAddGrad=^gradients_1/lstm/rnn/while/basic_lstm_cell/split_grad/concat
�
Pgradients_1/lstm/rnn/while/basic_lstm_cell/BiasAdd_grad/tuple/control_dependencyIdentity<gradients_1/lstm/rnn/while/basic_lstm_cell/split_grad/concatI^gradients_1/lstm/rnn/while/basic_lstm_cell/BiasAdd_grad/tuple/group_deps*
T0*O
_classE
CAloc:@gradients_1/lstm/rnn/while/basic_lstm_cell/split_grad/concat
�
Rgradients_1/lstm/rnn/while/basic_lstm_cell/BiasAdd_grad/tuple/control_dependency_1IdentityCgradients_1/lstm/rnn/while/basic_lstm_cell/BiasAdd_grad/BiasAddGradI^gradients_1/lstm/rnn/while/basic_lstm_cell/BiasAdd_grad/tuple/group_deps*
T0*V
_classL
JHloc:@gradients_1/lstm/rnn/while/basic_lstm_cell/BiasAdd_grad/BiasAddGrad
�
=gradients_1/lstm/rnn/while/basic_lstm_cell/MatMul_grad/MatMulMatMulPgradients_1/lstm/rnn/while/basic_lstm_cell/BiasAdd_grad/tuple/control_dependencyCgradients_1/lstm/rnn/while/basic_lstm_cell/MatMul_grad/MatMul/Enter*
transpose_b(*
T0*
transpose_a( 
�
Cgradients_1/lstm/rnn/while/basic_lstm_cell/MatMul_grad/MatMul/EnterEnter$lstm/rnn/basic_lstm_cell/kernel/read*
T0*
is_constant(*
parallel_iterations *8

frame_name*(gradients_1/lstm/rnn/while/while_context
�
?gradients_1/lstm/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1MatMulJgradients_1/lstm/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2Pgradients_1/lstm/rnn/while/basic_lstm_cell/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(
�
Egradients_1/lstm/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/ConstConst*8
_class.
,*loc:@lstm/rnn/while/basic_lstm_cell/concat*
valueB :
���������*
dtype0
�
Egradients_1/lstm/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/f_accStackV2Egradients_1/lstm/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/Const*
	elem_type0*8
_class.
,*loc:@lstm/rnn/while/basic_lstm_cell/concat*

stack_name 
�
Egradients_1/lstm/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/EnterEnterEgradients_1/lstm/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/f_acc*
T0*
is_constant(*
parallel_iterations *,

frame_namelstm/rnn/while/while_context
�
Kgradients_1/lstm/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/StackPushV2StackPushV2Egradients_1/lstm/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/Enter%lstm/rnn/while/basic_lstm_cell/concat^gradients_1/Add*
T0*
swap_memory( 
�
Jgradients_1/lstm/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2
StackPopV2Pgradients_1/lstm/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2/Enter^gradients_1/Sub*
	elem_type0
�
Pgradients_1/lstm/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/StackPopV2/EnterEnterEgradients_1/lstm/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1/f_acc*
T0*
is_constant(*
parallel_iterations *8

frame_name*(gradients_1/lstm/rnn/while/while_context
�
Ggradients_1/lstm/rnn/while/basic_lstm_cell/MatMul_grad/tuple/group_depsNoOp>^gradients_1/lstm/rnn/while/basic_lstm_cell/MatMul_grad/MatMul@^gradients_1/lstm/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1
�
Ogradients_1/lstm/rnn/while/basic_lstm_cell/MatMul_grad/tuple/control_dependencyIdentity=gradients_1/lstm/rnn/while/basic_lstm_cell/MatMul_grad/MatMulH^gradients_1/lstm/rnn/while/basic_lstm_cell/MatMul_grad/tuple/group_deps*
T0*P
_classF
DBloc:@gradients_1/lstm/rnn/while/basic_lstm_cell/MatMul_grad/MatMul
�
Qgradients_1/lstm/rnn/while/basic_lstm_cell/MatMul_grad/tuple/control_dependency_1Identity?gradients_1/lstm/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1H^gradients_1/lstm/rnn/while/basic_lstm_cell/MatMul_grad/tuple/group_deps*
T0*R
_classH
FDloc:@gradients_1/lstm/rnn/while/basic_lstm_cell/MatMul_grad/MatMul_1
u
Cgradients_1/lstm/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/b_accConst*
valueB�*    *
dtype0
�
Egradients_1/lstm/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_1EnterCgradients_1/lstm/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/b_acc*
T0*
is_constant( *
parallel_iterations *8

frame_name*(gradients_1/lstm/rnn/while/while_context
�
Egradients_1/lstm/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_2MergeEgradients_1/lstm/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_1Kgradients_1/lstm/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/NextIteration*
T0*
N
�
Dgradients_1/lstm/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/SwitchSwitchEgradients_1/lstm/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_2gradients_1/b_count_2*
T0
�
Agradients_1/lstm/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/AddAddFgradients_1/lstm/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/Switch:1Rgradients_1/lstm/rnn/while/basic_lstm_cell/BiasAdd_grad/tuple/control_dependency_1*
T0
�
Kgradients_1/lstm/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/NextIterationNextIterationAgradients_1/lstm/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/Add*
T0
�
Egradients_1/lstm/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_3ExitDgradients_1/lstm/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/Switch*
T0
x
<gradients_1/lstm/rnn/while/basic_lstm_cell/concat_grad/ConstConst^gradients_1/Sub*
value	B :*
dtype0
w
;gradients_1/lstm/rnn/while/basic_lstm_cell/concat_grad/RankConst^gradients_1/Sub*
value	B :*
dtype0
�
:gradients_1/lstm/rnn/while/basic_lstm_cell/concat_grad/modFloorMod<gradients_1/lstm/rnn/while/basic_lstm_cell/concat_grad/Const;gradients_1/lstm/rnn/while/basic_lstm_cell/concat_grad/Rank*
T0
�
<gradients_1/lstm/rnn/while/basic_lstm_cell/concat_grad/ShapeShape lstm/rnn/while/TensorArrayReadV3*
T0*
out_type0
�
=gradients_1/lstm/rnn/while/basic_lstm_cell/concat_grad/ShapeNShapeNHgradients_1/lstm/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPopV2Jgradients_1/lstm/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPopV2_1*
T0*
out_type0*
N
�
Cgradients_1/lstm/rnn/while/basic_lstm_cell/concat_grad/ShapeN/ConstConst*3
_class)
'%loc:@lstm/rnn/while/TensorArrayReadV3*
valueB :
���������*
dtype0
�
Cgradients_1/lstm/rnn/while/basic_lstm_cell/concat_grad/ShapeN/f_accStackV2Cgradients_1/lstm/rnn/while/basic_lstm_cell/concat_grad/ShapeN/Const*
	elem_type0*3
_class)
'%loc:@lstm/rnn/while/TensorArrayReadV3*

stack_name 
�
Cgradients_1/lstm/rnn/while/basic_lstm_cell/concat_grad/ShapeN/EnterEnterCgradients_1/lstm/rnn/while/basic_lstm_cell/concat_grad/ShapeN/f_acc*
T0*
is_constant(*
parallel_iterations *,

frame_namelstm/rnn/while/while_context
�
Igradients_1/lstm/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPushV2StackPushV2Cgradients_1/lstm/rnn/while/basic_lstm_cell/concat_grad/ShapeN/Enter lstm/rnn/while/TensorArrayReadV3^gradients_1/Add*
T0*
swap_memory( 
�
Hgradients_1/lstm/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPopV2
StackPopV2Ngradients_1/lstm/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPopV2/Enter^gradients_1/Sub*
	elem_type0
�
Ngradients_1/lstm/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPopV2/EnterEnterCgradients_1/lstm/rnn/while/basic_lstm_cell/concat_grad/ShapeN/f_acc*
T0*
is_constant(*
parallel_iterations *8

frame_name*(gradients_1/lstm/rnn/while/while_context
�
Egradients_1/lstm/rnn/while/basic_lstm_cell/concat_grad/ShapeN/Const_1Const*,
_class"
 loc:@lstm/rnn/while/Identity_4*
valueB :
���������*
dtype0
�
Egradients_1/lstm/rnn/while/basic_lstm_cell/concat_grad/ShapeN/f_acc_1StackV2Egradients_1/lstm/rnn/while/basic_lstm_cell/concat_grad/ShapeN/Const_1*
	elem_type0*,
_class"
 loc:@lstm/rnn/while/Identity_4*

stack_name 
�
Egradients_1/lstm/rnn/while/basic_lstm_cell/concat_grad/ShapeN/Enter_1EnterEgradients_1/lstm/rnn/while/basic_lstm_cell/concat_grad/ShapeN/f_acc_1*
T0*
is_constant(*
parallel_iterations *,

frame_namelstm/rnn/while/while_context
�
Kgradients_1/lstm/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPushV2_1StackPushV2Egradients_1/lstm/rnn/while/basic_lstm_cell/concat_grad/ShapeN/Enter_1lstm/rnn/while/Identity_4^gradients_1/Add*
T0*
swap_memory( 
�
Jgradients_1/lstm/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPopV2_1
StackPopV2Pgradients_1/lstm/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPopV2_1/Enter^gradients_1/Sub*
	elem_type0
�
Pgradients_1/lstm/rnn/while/basic_lstm_cell/concat_grad/ShapeN/StackPopV2_1/EnterEnterEgradients_1/lstm/rnn/while/basic_lstm_cell/concat_grad/ShapeN/f_acc_1*
T0*
is_constant(*
parallel_iterations *8

frame_name*(gradients_1/lstm/rnn/while/while_context
�
Cgradients_1/lstm/rnn/while/basic_lstm_cell/concat_grad/ConcatOffsetConcatOffset:gradients_1/lstm/rnn/while/basic_lstm_cell/concat_grad/mod=gradients_1/lstm/rnn/while/basic_lstm_cell/concat_grad/ShapeN?gradients_1/lstm/rnn/while/basic_lstm_cell/concat_grad/ShapeN:1*
N
�
<gradients_1/lstm/rnn/while/basic_lstm_cell/concat_grad/SliceSliceOgradients_1/lstm/rnn/while/basic_lstm_cell/MatMul_grad/tuple/control_dependencyCgradients_1/lstm/rnn/while/basic_lstm_cell/concat_grad/ConcatOffset=gradients_1/lstm/rnn/while/basic_lstm_cell/concat_grad/ShapeN*
T0*
Index0
�
>gradients_1/lstm/rnn/while/basic_lstm_cell/concat_grad/Slice_1SliceOgradients_1/lstm/rnn/while/basic_lstm_cell/MatMul_grad/tuple/control_dependencyEgradients_1/lstm/rnn/while/basic_lstm_cell/concat_grad/ConcatOffset:1?gradients_1/lstm/rnn/while/basic_lstm_cell/concat_grad/ShapeN:1*
T0*
Index0
�
Ggradients_1/lstm/rnn/while/basic_lstm_cell/concat_grad/tuple/group_depsNoOp=^gradients_1/lstm/rnn/while/basic_lstm_cell/concat_grad/Slice?^gradients_1/lstm/rnn/while/basic_lstm_cell/concat_grad/Slice_1
�
Ogradients_1/lstm/rnn/while/basic_lstm_cell/concat_grad/tuple/control_dependencyIdentity<gradients_1/lstm/rnn/while/basic_lstm_cell/concat_grad/SliceH^gradients_1/lstm/rnn/while/basic_lstm_cell/concat_grad/tuple/group_deps*
T0*O
_classE
CAloc:@gradients_1/lstm/rnn/while/basic_lstm_cell/concat_grad/Slice
�
Qgradients_1/lstm/rnn/while/basic_lstm_cell/concat_grad/tuple/control_dependency_1Identity>gradients_1/lstm/rnn/while/basic_lstm_cell/concat_grad/Slice_1H^gradients_1/lstm/rnn/while/basic_lstm_cell/concat_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients_1/lstm/rnn/while/basic_lstm_cell/concat_grad/Slice_1
y
Bgradients_1/lstm/rnn/while/basic_lstm_cell/MatMul/Enter_grad/b_accConst*
valueB
��*    *
dtype0
�
Dgradients_1/lstm/rnn/while/basic_lstm_cell/MatMul/Enter_grad/b_acc_1EnterBgradients_1/lstm/rnn/while/basic_lstm_cell/MatMul/Enter_grad/b_acc*
T0*
is_constant( *
parallel_iterations *8

frame_name*(gradients_1/lstm/rnn/while/while_context
�
Dgradients_1/lstm/rnn/while/basic_lstm_cell/MatMul/Enter_grad/b_acc_2MergeDgradients_1/lstm/rnn/while/basic_lstm_cell/MatMul/Enter_grad/b_acc_1Jgradients_1/lstm/rnn/while/basic_lstm_cell/MatMul/Enter_grad/NextIteration*
T0*
N
�
Cgradients_1/lstm/rnn/while/basic_lstm_cell/MatMul/Enter_grad/SwitchSwitchDgradients_1/lstm/rnn/while/basic_lstm_cell/MatMul/Enter_grad/b_acc_2gradients_1/b_count_2*
T0
�
@gradients_1/lstm/rnn/while/basic_lstm_cell/MatMul/Enter_grad/AddAddEgradients_1/lstm/rnn/while/basic_lstm_cell/MatMul/Enter_grad/Switch:1Qgradients_1/lstm/rnn/while/basic_lstm_cell/MatMul_grad/tuple/control_dependency_1*
T0
�
Jgradients_1/lstm/rnn/while/basic_lstm_cell/MatMul/Enter_grad/NextIterationNextIteration@gradients_1/lstm/rnn/while/basic_lstm_cell/MatMul/Enter_grad/Add*
T0
�
Dgradients_1/lstm/rnn/while/basic_lstm_cell/MatMul/Enter_grad/b_acc_3ExitCgradients_1/lstm/rnn/while/basic_lstm_cell/MatMul/Enter_grad/Switch*
T0
�
Sgradients_1/lstm/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3Ygradients_1/lstm/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter[gradients_1/lstm/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1^gradients_1/Sub*9
_class/
-+loc:@lstm/rnn/while/TensorArrayReadV3/Enter*
sourcegradients_1
�
Ygradients_1/lstm/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterEnterlstm/rnn/TensorArray_1*
T0*9
_class/
-+loc:@lstm/rnn/while/TensorArrayReadV3/Enter*
is_constant(*
parallel_iterations *8

frame_name*(gradients_1/lstm/rnn/while/while_context
�
[gradients_1/lstm/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1EnterClstm/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
T0*9
_class/
-+loc:@lstm/rnn/while/TensorArrayReadV3/Enter*
is_constant(*
parallel_iterations *8

frame_name*(gradients_1/lstm/rnn/while/while_context
�
Ogradients_1/lstm/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/gradient_flowIdentity[gradients_1/lstm/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1T^gradients_1/lstm/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*9
_class/
-+loc:@lstm/rnn/while/TensorArrayReadV3/Enter
�
Ugradients_1/lstm/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3Sgradients_1/lstm/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3`gradients_1/lstm/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2Ogradients_1/lstm/rnn/while/basic_lstm_cell/concat_grad/tuple/control_dependencyOgradients_1/lstm/rnn/while/TensorArrayReadV3_grad/TensorArrayGrad/gradient_flow*
T0
l
?gradients_1/lstm/rnn/while/TensorArrayReadV3/Enter_1_grad/b_accConst*
valueB
 *    *
dtype0
�
Agradients_1/lstm/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_1Enter?gradients_1/lstm/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc*
T0*
is_constant( *
parallel_iterations *8

frame_name*(gradients_1/lstm/rnn/while/while_context
�
Agradients_1/lstm/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_2MergeAgradients_1/lstm/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_1Ggradients_1/lstm/rnn/while/TensorArrayReadV3/Enter_1_grad/NextIteration*
T0*
N
�
@gradients_1/lstm/rnn/while/TensorArrayReadV3/Enter_1_grad/SwitchSwitchAgradients_1/lstm/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_2gradients_1/b_count_2*
T0
�
=gradients_1/lstm/rnn/while/TensorArrayReadV3/Enter_1_grad/AddAddBgradients_1/lstm/rnn/while/TensorArrayReadV3/Enter_1_grad/Switch:1Ugradients_1/lstm/rnn/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3*
T0
�
Ggradients_1/lstm/rnn/while/TensorArrayReadV3/Enter_1_grad/NextIterationNextIteration=gradients_1/lstm/rnn/while/TensorArrayReadV3/Enter_1_grad/Add*
T0
�
Agradients_1/lstm/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3Exit@gradients_1/lstm/rnn/while/TensorArrayReadV3/Enter_1_grad/Switch*
T0
�
8gradients_1/lstm/rnn/while/Switch_4_grad_1/NextIterationNextIterationQgradients_1/lstm/rnn/while/basic_lstm_cell/concat_grad/tuple/control_dependency_1*
T0
�
vgradients_1/lstm/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3lstm/rnn/TensorArray_1Agradients_1/lstm/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3*)
_class
loc:@lstm/rnn/TensorArray_1*
sourcegradients_1
�
rgradients_1/lstm/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/gradient_flowIdentityAgradients_1/lstm/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3w^gradients_1/lstm/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*)
_class
loc:@lstm/rnn/TensorArray_1
�
hgradients_1/lstm/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3TensorArrayGatherV3vgradients_1/lstm/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3!lstm/rnn/TensorArrayUnstack/rangergradients_1/lstm/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/gradient_flow*
element_shape:*
dtype0
�
egradients_1/lstm/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/group_depsNoOpi^gradients_1/lstm/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3B^gradients_1/lstm/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3
�
mgradients_1/lstm/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/control_dependencyIdentityhgradients_1/lstm/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3f^gradients_1/lstm/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/group_deps*
T0*{
_classq
omloc:@gradients_1/lstm/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3
�
ogradients_1/lstm/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/control_dependency_1IdentityAgradients_1/lstm/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3f^gradients_1/lstm/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/group_deps*
T0*T
_classJ
HFloc:@gradients_1/lstm/rnn/while/TensorArrayReadV3/Enter_1_grad/b_acc_3
d
5gradients_1/lstm/rnn/transpose_grad/InvertPermutationInvertPermutationlstm/rnn/concat*
T0
�
-gradients_1/lstm/rnn/transpose_grad/transpose	Transposemgradients_1/lstm/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/control_dependency5gradients_1/lstm/rnn/transpose_grad/InvertPermutation*
Tperm0*
T0
J
gradients_1/Reshape_grad/ShapeShapeconcat_2*
T0*
out_type0
�
 gradients_1/Reshape_grad/ReshapeReshape-gradients_1/lstm/rnn/transpose_grad/transposegradients_1/Reshape_grad/Shape*
T0*
Tshape0
H
gradients_1/concat_2_grad/RankConst*
value	B :*
dtype0
a
gradients_1/concat_2_grad/modFloorModconcat_2/axisgradients_1/concat_2_grad/Rank*
T0
P
gradients_1/concat_2_grad/ShapeShapeconcat/concat*
T0*
out_type0
l
 gradients_1/concat_2_grad/ShapeNShapeNconcat/concatconcat_1/concat*
T0*
out_type0*
N
�
&gradients_1/concat_2_grad/ConcatOffsetConcatOffsetgradients_1/concat_2_grad/mod gradients_1/concat_2_grad/ShapeN"gradients_1/concat_2_grad/ShapeN:1*
N
�
gradients_1/concat_2_grad/SliceSlice gradients_1/Reshape_grad/Reshape&gradients_1/concat_2_grad/ConcatOffset gradients_1/concat_2_grad/ShapeN*
T0*
Index0
�
!gradients_1/concat_2_grad/Slice_1Slice gradients_1/Reshape_grad/Reshape(gradients_1/concat_2_grad/ConcatOffset:1"gradients_1/concat_2_grad/ShapeN:1*
T0*
Index0
x
*gradients_1/concat_2_grad/tuple/group_depsNoOp ^gradients_1/concat_2_grad/Slice"^gradients_1/concat_2_grad/Slice_1
�
2gradients_1/concat_2_grad/tuple/control_dependencyIdentitygradients_1/concat_2_grad/Slice+^gradients_1/concat_2_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients_1/concat_2_grad/Slice
�
4gradients_1/concat_2_grad/tuple/control_dependency_1Identity!gradients_1/concat_2_grad/Slice_1+^gradients_1/concat_2_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients_1/concat_2_grad/Slice_1
�
]gradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/Mul_grad/ShapeShapeJmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/BiasAdd*
T0*
out_type0
�
_gradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/Mul_grad/Shape_1ShapeJmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/Sigmoid*
T0*
out_type0
�
mgradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/Mul_grad/BroadcastGradientArgsBroadcastGradientArgs]gradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/Mul_grad/Shape_gradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/Mul_grad/Shape_1*
T0
�
[gradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/Mul_grad/MulMul2gradients_1/concat_2_grad/tuple/control_dependencyJmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/Sigmoid*
T0
�
[gradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/Mul_grad/SumSum[gradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/Mul_grad/Mulmgradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/Mul_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
�
_gradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/Mul_grad/ReshapeReshape[gradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/Mul_grad/Sum]gradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/Mul_grad/Shape*
T0*
Tshape0
�
]gradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/Mul_grad/Mul_1MulJmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/BiasAdd2gradients_1/concat_2_grad/tuple/control_dependency*
T0
�
]gradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/Mul_grad/Sum_1Sum]gradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/Mul_grad/Mul_1ogradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/Mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
�
agradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/Mul_grad/Reshape_1Reshape]gradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/Mul_grad/Sum_1_gradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/Mul_grad/Shape_1*
T0*
Tshape0
�
hgradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/Mul_grad/tuple/group_depsNoOp`^gradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/Mul_grad/Reshapeb^gradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/Mul_grad/Reshape_1
�
pgradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/Mul_grad/tuple/control_dependencyIdentity_gradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/Mul_grad/Reshapei^gradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/Mul_grad/tuple/group_deps*
T0*r
_classh
fdloc:@gradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/Mul_grad/Reshape
�
rgradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/Mul_grad/tuple/control_dependency_1Identityagradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/Mul_grad/Reshape_1i^gradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/Mul_grad/tuple/group_deps*
T0*t
_classj
hfloc:@gradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/Mul_grad/Reshape_1
�
ggradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/Sigmoid_grad/SigmoidGradSigmoidGradJmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/Sigmoidrgradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/Mul_grad/tuple/control_dependency_1*
T0
�
gradients_1/AddN_7AddNpgradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/Mul_grad/tuple/control_dependencyggradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/Sigmoid_grad/SigmoidGrad*
T0*r
_classh
fdloc:@gradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/Mul_grad/Reshape*
N
�
ggradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/BiasAdd_grad/BiasAddGradBiasAddGradgradients_1/AddN_7*
T0*
data_formatNHWC
�
lgradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/BiasAdd_grad/tuple/group_depsNoOp^gradients_1/AddN_7h^gradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/BiasAdd_grad/BiasAddGrad
�
tgradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/BiasAdd_grad/tuple/control_dependencyIdentitygradients_1/AddN_7m^gradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/BiasAdd_grad/tuple/group_deps*
T0*r
_classh
fdloc:@gradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/Mul_grad/Reshape
�
vgradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/BiasAdd_grad/tuple/control_dependency_1Identityggradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/BiasAdd_grad/BiasAddGradm^gradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/BiasAdd_grad/tuple/group_deps*
T0*z
_classp
nlloc:@gradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/BiasAdd_grad/BiasAddGrad
�
agradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/MatMul_grad/MatMulMatMultgradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/BiasAdd_grad/tuple/control_dependencyNmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/kernel/read*
transpose_b(*
T0*
transpose_a( 
�
cgradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/MatMul_grad/MatMul_1MatMulFmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/Multgradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(
�
kgradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/MatMul_grad/tuple/group_depsNoOpb^gradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/MatMul_grad/MatMuld^gradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/MatMul_grad/MatMul_1
�
sgradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/MatMul_grad/tuple/control_dependencyIdentityagradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/MatMul_grad/MatMull^gradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/MatMul_grad/tuple/group_deps*
T0*t
_classj
hfloc:@gradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/MatMul_grad/MatMul
�
ugradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/MatMul_grad/tuple/control_dependency_1Identitycgradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/MatMul_grad/MatMul_1l^gradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/MatMul_grad/tuple/group_deps*
T0*v
_classl
jhloc:@gradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/MatMul_grad/MatMul_1
�
]gradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/Mul_grad/ShapeShapeJmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/BiasAdd*
T0*
out_type0
�
_gradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/Mul_grad/Shape_1ShapeJmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/Sigmoid*
T0*
out_type0
�
mgradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/Mul_grad/BroadcastGradientArgsBroadcastGradientArgs]gradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/Mul_grad/Shape_gradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/Mul_grad/Shape_1*
T0
�
[gradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/Mul_grad/MulMulsgradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/MatMul_grad/tuple/control_dependencyJmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/Sigmoid*
T0
�
[gradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/Mul_grad/SumSum[gradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/Mul_grad/Mulmgradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/Mul_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0
�
_gradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/Mul_grad/ReshapeReshape[gradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/Mul_grad/Sum]gradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/Mul_grad/Shape*
T0*
Tshape0
�
]gradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/Mul_grad/Mul_1MulJmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/BiasAddsgradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/MatMul_grad/tuple/control_dependency*
T0
�
]gradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/Mul_grad/Sum_1Sum]gradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/Mul_grad/Mul_1ogradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/Mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0
�
agradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/Mul_grad/Reshape_1Reshape]gradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/Mul_grad/Sum_1_gradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/Mul_grad/Shape_1*
T0*
Tshape0
�
hgradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/Mul_grad/tuple/group_depsNoOp`^gradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/Mul_grad/Reshapeb^gradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/Mul_grad/Reshape_1
�
pgradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/Mul_grad/tuple/control_dependencyIdentity_gradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/Mul_grad/Reshapei^gradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/Mul_grad/tuple/group_deps*
T0*r
_classh
fdloc:@gradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/Mul_grad/Reshape
�
rgradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/Mul_grad/tuple/control_dependency_1Identityagradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/Mul_grad/Reshape_1i^gradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/Mul_grad/tuple/group_deps*
T0*t
_classj
hfloc:@gradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/Mul_grad/Reshape_1
�
ggradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/Sigmoid_grad/SigmoidGradSigmoidGradJmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/Sigmoidrgradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/Mul_grad/tuple/control_dependency_1*
T0
�
gradients_1/AddN_8AddNpgradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/Mul_grad/tuple/control_dependencyggradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/Sigmoid_grad/SigmoidGrad*
T0*r
_classh
fdloc:@gradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/Mul_grad/Reshape*
N
�
ggradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/BiasAdd_grad/BiasAddGradBiasAddGradgradients_1/AddN_8*
T0*
data_formatNHWC
�
lgradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/BiasAdd_grad/tuple/group_depsNoOp^gradients_1/AddN_8h^gradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/BiasAdd_grad/BiasAddGrad
�
tgradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/BiasAdd_grad/tuple/control_dependencyIdentitygradients_1/AddN_8m^gradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/BiasAdd_grad/tuple/group_deps*
T0*r
_classh
fdloc:@gradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/Mul_grad/Reshape
�
vgradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/BiasAdd_grad/tuple/control_dependency_1Identityggradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/BiasAdd_grad/BiasAddGradm^gradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/BiasAdd_grad/tuple/group_deps*
T0*z
_classp
nlloc:@gradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/BiasAdd_grad/BiasAddGrad
�
agradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/MatMul_grad/MatMulMatMultgradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/BiasAdd_grad/tuple/control_dependencyNmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/kernel/read*
transpose_b(*
T0*
transpose_a( 
�
cgradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/MatMul_grad/MatMul_1MatMul-main_graph_0_encoder0/Flatten/flatten/Reshapetgradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(
�
kgradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/MatMul_grad/tuple/group_depsNoOpb^gradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/MatMul_grad/MatMuld^gradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/MatMul_grad/MatMul_1
�
sgradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/MatMul_grad/tuple/control_dependencyIdentityagradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/MatMul_grad/MatMull^gradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/MatMul_grad/tuple/group_deps*
T0*t
_classj
hfloc:@gradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/MatMul_grad/MatMul
�
ugradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/MatMul_grad/tuple/control_dependency_1Identitycgradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/MatMul_grad/MatMul_1l^gradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/MatMul_grad/tuple/group_deps*
T0*v
_classl
jhloc:@gradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/MatMul_grad/MatMul_1
�
Dgradients_1/main_graph_0_encoder0/Flatten/flatten/Reshape_grad/ShapeShape main_graph_0_encoder0/conv_1/Elu*
T0*
out_type0
�
Fgradients_1/main_graph_0_encoder0/Flatten/flatten/Reshape_grad/ReshapeReshapesgradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/MatMul_grad/tuple/control_dependencyDgradients_1/main_graph_0_encoder0/Flatten/flatten/Reshape_grad/Shape*
T0*
Tshape0
�
9gradients_1/main_graph_0_encoder0/conv_1/Elu_grad/EluGradEluGradFgradients_1/main_graph_0_encoder0/Flatten/flatten/Reshape_grad/Reshape main_graph_0_encoder0/conv_1/Elu*
T0
�
Agradients_1/main_graph_0_encoder0/conv_1/BiasAdd_grad/BiasAddGradBiasAddGrad9gradients_1/main_graph_0_encoder0/conv_1/Elu_grad/EluGrad*
T0*
data_formatNHWC
�
Fgradients_1/main_graph_0_encoder0/conv_1/BiasAdd_grad/tuple/group_depsNoOpB^gradients_1/main_graph_0_encoder0/conv_1/BiasAdd_grad/BiasAddGrad:^gradients_1/main_graph_0_encoder0/conv_1/Elu_grad/EluGrad
�
Ngradients_1/main_graph_0_encoder0/conv_1/BiasAdd_grad/tuple/control_dependencyIdentity9gradients_1/main_graph_0_encoder0/conv_1/Elu_grad/EluGradG^gradients_1/main_graph_0_encoder0/conv_1/BiasAdd_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients_1/main_graph_0_encoder0/conv_1/Elu_grad/EluGrad
�
Pgradients_1/main_graph_0_encoder0/conv_1/BiasAdd_grad/tuple/control_dependency_1IdentityAgradients_1/main_graph_0_encoder0/conv_1/BiasAdd_grad/BiasAddGradG^gradients_1/main_graph_0_encoder0/conv_1/BiasAdd_grad/tuple/group_deps*
T0*T
_classJ
HFloc:@gradients_1/main_graph_0_encoder0/conv_1/BiasAdd_grad/BiasAddGrad
�
;gradients_1/main_graph_0_encoder0/conv_1/Conv2D_grad/ShapeNShapeN main_graph_0_encoder0/conv_0/Elu(main_graph_0_encoder0/conv_1/kernel/read*
T0*
out_type0*
N
�
Hgradients_1/main_graph_0_encoder0/conv_1/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput;gradients_1/main_graph_0_encoder0/conv_1/Conv2D_grad/ShapeN(main_graph_0_encoder0/conv_1/kernel/readNgradients_1/main_graph_0_encoder0/conv_1/BiasAdd_grad/tuple/control_dependency*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingVALID
�
Igradients_1/main_graph_0_encoder0/conv_1/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFilter main_graph_0_encoder0/conv_0/Elu=gradients_1/main_graph_0_encoder0/conv_1/Conv2D_grad/ShapeN:1Ngradients_1/main_graph_0_encoder0/conv_1/BiasAdd_grad/tuple/control_dependency*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingVALID
�
Egradients_1/main_graph_0_encoder0/conv_1/Conv2D_grad/tuple/group_depsNoOpJ^gradients_1/main_graph_0_encoder0/conv_1/Conv2D_grad/Conv2DBackpropFilterI^gradients_1/main_graph_0_encoder0/conv_1/Conv2D_grad/Conv2DBackpropInput
�
Mgradients_1/main_graph_0_encoder0/conv_1/Conv2D_grad/tuple/control_dependencyIdentityHgradients_1/main_graph_0_encoder0/conv_1/Conv2D_grad/Conv2DBackpropInputF^gradients_1/main_graph_0_encoder0/conv_1/Conv2D_grad/tuple/group_deps*
T0*[
_classQ
OMloc:@gradients_1/main_graph_0_encoder0/conv_1/Conv2D_grad/Conv2DBackpropInput
�
Ogradients_1/main_graph_0_encoder0/conv_1/Conv2D_grad/tuple/control_dependency_1IdentityIgradients_1/main_graph_0_encoder0/conv_1/Conv2D_grad/Conv2DBackpropFilterF^gradients_1/main_graph_0_encoder0/conv_1/Conv2D_grad/tuple/group_deps*
T0*\
_classR
PNloc:@gradients_1/main_graph_0_encoder0/conv_1/Conv2D_grad/Conv2DBackpropFilter
�
9gradients_1/main_graph_0_encoder0/conv_0/Elu_grad/EluGradEluGradMgradients_1/main_graph_0_encoder0/conv_1/Conv2D_grad/tuple/control_dependency main_graph_0_encoder0/conv_0/Elu*
T0
�
Agradients_1/main_graph_0_encoder0/conv_0/BiasAdd_grad/BiasAddGradBiasAddGrad9gradients_1/main_graph_0_encoder0/conv_0/Elu_grad/EluGrad*
T0*
data_formatNHWC
�
Fgradients_1/main_graph_0_encoder0/conv_0/BiasAdd_grad/tuple/group_depsNoOpB^gradients_1/main_graph_0_encoder0/conv_0/BiasAdd_grad/BiasAddGrad:^gradients_1/main_graph_0_encoder0/conv_0/Elu_grad/EluGrad
�
Ngradients_1/main_graph_0_encoder0/conv_0/BiasAdd_grad/tuple/control_dependencyIdentity9gradients_1/main_graph_0_encoder0/conv_0/Elu_grad/EluGradG^gradients_1/main_graph_0_encoder0/conv_0/BiasAdd_grad/tuple/group_deps*
T0*L
_classB
@>loc:@gradients_1/main_graph_0_encoder0/conv_0/Elu_grad/EluGrad
�
Pgradients_1/main_graph_0_encoder0/conv_0/BiasAdd_grad/tuple/control_dependency_1IdentityAgradients_1/main_graph_0_encoder0/conv_0/BiasAdd_grad/BiasAddGradG^gradients_1/main_graph_0_encoder0/conv_0/BiasAdd_grad/tuple/group_deps*
T0*T
_classJ
HFloc:@gradients_1/main_graph_0_encoder0/conv_0/BiasAdd_grad/BiasAddGrad
�
;gradients_1/main_graph_0_encoder0/conv_0/Conv2D_grad/ShapeNShapeNvisual_observation_0(main_graph_0_encoder0/conv_0/kernel/read*
T0*
out_type0*
N
�
Hgradients_1/main_graph_0_encoder0/conv_0/Conv2D_grad/Conv2DBackpropInputConv2DBackpropInput;gradients_1/main_graph_0_encoder0/conv_0/Conv2D_grad/ShapeN(main_graph_0_encoder0/conv_0/kernel/readNgradients_1/main_graph_0_encoder0/conv_0/BiasAdd_grad/tuple/control_dependency*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingVALID
�
Igradients_1/main_graph_0_encoder0/conv_0/Conv2D_grad/Conv2DBackpropFilterConv2DBackpropFiltervisual_observation_0=gradients_1/main_graph_0_encoder0/conv_0/Conv2D_grad/ShapeN:1Ngradients_1/main_graph_0_encoder0/conv_0/BiasAdd_grad/tuple/control_dependency*
	dilations
*
T0*
data_formatNHWC*
strides
*
explicit_paddings
 *
use_cudnn_on_gpu(*
paddingVALID
�
Egradients_1/main_graph_0_encoder0/conv_0/Conv2D_grad/tuple/group_depsNoOpJ^gradients_1/main_graph_0_encoder0/conv_0/Conv2D_grad/Conv2DBackpropFilterI^gradients_1/main_graph_0_encoder0/conv_0/Conv2D_grad/Conv2DBackpropInput
�
Mgradients_1/main_graph_0_encoder0/conv_0/Conv2D_grad/tuple/control_dependencyIdentityHgradients_1/main_graph_0_encoder0/conv_0/Conv2D_grad/Conv2DBackpropInputF^gradients_1/main_graph_0_encoder0/conv_0/Conv2D_grad/tuple/group_deps*
T0*[
_classQ
OMloc:@gradients_1/main_graph_0_encoder0/conv_0/Conv2D_grad/Conv2DBackpropInput
�
Ogradients_1/main_graph_0_encoder0/conv_0/Conv2D_grad/tuple/control_dependency_1IdentityIgradients_1/main_graph_0_encoder0/conv_0/Conv2D_grad/Conv2DBackpropFilterF^gradients_1/main_graph_0_encoder0/conv_0/Conv2D_grad/tuple/group_deps*
T0*\
_classR
PNloc:@gradients_1/main_graph_0_encoder0/conv_0/Conv2D_grad/Conv2DBackpropFilter
g
beta1_power/initial_valueConst*
_class
loc:@dense/kernel*
valueB
 *fff?*
dtype0
x
beta1_power
VariableV2*
shape: *
shared_name *
_class
loc:@dense/kernel*
dtype0*
	container 
�
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
use_locking(*
T0*
_class
loc:@dense/kernel*
validate_shape(
S
beta1_power/readIdentitybeta1_power*
T0*
_class
loc:@dense/kernel
g
beta2_power/initial_valueConst*
_class
loc:@dense/kernel*
valueB
 *w�?*
dtype0
x
beta2_power
VariableV2*
shape: *
shared_name *
_class
loc:@dense/kernel*
dtype0*
	container 
�
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
use_locking(*
T0*
_class
loc:@dense/kernel*
validate_shape(
S
beta2_power/readIdentitybeta2_power*
T0*
_class
loc:@dense/kernel
�
Jmain_graph_0_encoder0/conv_0/kernel/Adam/Initializer/zeros/shape_as_tensorConst*%
valueB"             *6
_class,
*(loc:@main_graph_0_encoder0/conv_0/kernel*
dtype0
�
@main_graph_0_encoder0/conv_0/kernel/Adam/Initializer/zeros/ConstConst*
valueB
 *    *6
_class,
*(loc:@main_graph_0_encoder0/conv_0/kernel*
dtype0
�
:main_graph_0_encoder0/conv_0/kernel/Adam/Initializer/zerosFillJmain_graph_0_encoder0/conv_0/kernel/Adam/Initializer/zeros/shape_as_tensor@main_graph_0_encoder0/conv_0/kernel/Adam/Initializer/zeros/Const*
T0*

index_type0*6
_class,
*(loc:@main_graph_0_encoder0/conv_0/kernel
�
(main_graph_0_encoder0/conv_0/kernel/Adam
VariableV2*
shape: *
shared_name *6
_class,
*(loc:@main_graph_0_encoder0/conv_0/kernel*
dtype0*
	container 
�
/main_graph_0_encoder0/conv_0/kernel/Adam/AssignAssign(main_graph_0_encoder0/conv_0/kernel/Adam:main_graph_0_encoder0/conv_0/kernel/Adam/Initializer/zeros*
use_locking(*
T0*6
_class,
*(loc:@main_graph_0_encoder0/conv_0/kernel*
validate_shape(
�
-main_graph_0_encoder0/conv_0/kernel/Adam/readIdentity(main_graph_0_encoder0/conv_0/kernel/Adam*
T0*6
_class,
*(loc:@main_graph_0_encoder0/conv_0/kernel
�
Lmain_graph_0_encoder0/conv_0/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*%
valueB"             *6
_class,
*(loc:@main_graph_0_encoder0/conv_0/kernel*
dtype0
�
Bmain_graph_0_encoder0/conv_0/kernel/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *6
_class,
*(loc:@main_graph_0_encoder0/conv_0/kernel*
dtype0
�
<main_graph_0_encoder0/conv_0/kernel/Adam_1/Initializer/zerosFillLmain_graph_0_encoder0/conv_0/kernel/Adam_1/Initializer/zeros/shape_as_tensorBmain_graph_0_encoder0/conv_0/kernel/Adam_1/Initializer/zeros/Const*
T0*

index_type0*6
_class,
*(loc:@main_graph_0_encoder0/conv_0/kernel
�
*main_graph_0_encoder0/conv_0/kernel/Adam_1
VariableV2*
shape: *
shared_name *6
_class,
*(loc:@main_graph_0_encoder0/conv_0/kernel*
dtype0*
	container 
�
1main_graph_0_encoder0/conv_0/kernel/Adam_1/AssignAssign*main_graph_0_encoder0/conv_0/kernel/Adam_1<main_graph_0_encoder0/conv_0/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*6
_class,
*(loc:@main_graph_0_encoder0/conv_0/kernel*
validate_shape(
�
/main_graph_0_encoder0/conv_0/kernel/Adam_1/readIdentity*main_graph_0_encoder0/conv_0/kernel/Adam_1*
T0*6
_class,
*(loc:@main_graph_0_encoder0/conv_0/kernel
�
8main_graph_0_encoder0/conv_0/bias/Adam/Initializer/zerosConst*
valueB *    *4
_class*
(&loc:@main_graph_0_encoder0/conv_0/bias*
dtype0
�
&main_graph_0_encoder0/conv_0/bias/Adam
VariableV2*
shape: *
shared_name *4
_class*
(&loc:@main_graph_0_encoder0/conv_0/bias*
dtype0*
	container 
�
-main_graph_0_encoder0/conv_0/bias/Adam/AssignAssign&main_graph_0_encoder0/conv_0/bias/Adam8main_graph_0_encoder0/conv_0/bias/Adam/Initializer/zeros*
use_locking(*
T0*4
_class*
(&loc:@main_graph_0_encoder0/conv_0/bias*
validate_shape(
�
+main_graph_0_encoder0/conv_0/bias/Adam/readIdentity&main_graph_0_encoder0/conv_0/bias/Adam*
T0*4
_class*
(&loc:@main_graph_0_encoder0/conv_0/bias
�
:main_graph_0_encoder0/conv_0/bias/Adam_1/Initializer/zerosConst*
valueB *    *4
_class*
(&loc:@main_graph_0_encoder0/conv_0/bias*
dtype0
�
(main_graph_0_encoder0/conv_0/bias/Adam_1
VariableV2*
shape: *
shared_name *4
_class*
(&loc:@main_graph_0_encoder0/conv_0/bias*
dtype0*
	container 
�
/main_graph_0_encoder0/conv_0/bias/Adam_1/AssignAssign(main_graph_0_encoder0/conv_0/bias/Adam_1:main_graph_0_encoder0/conv_0/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*4
_class*
(&loc:@main_graph_0_encoder0/conv_0/bias*
validate_shape(
�
-main_graph_0_encoder0/conv_0/bias/Adam_1/readIdentity(main_graph_0_encoder0/conv_0/bias/Adam_1*
T0*4
_class*
(&loc:@main_graph_0_encoder0/conv_0/bias
�
Jmain_graph_0_encoder0/conv_1/kernel/Adam/Initializer/zeros/shape_as_tensorConst*%
valueB"              *6
_class,
*(loc:@main_graph_0_encoder0/conv_1/kernel*
dtype0
�
@main_graph_0_encoder0/conv_1/kernel/Adam/Initializer/zeros/ConstConst*
valueB
 *    *6
_class,
*(loc:@main_graph_0_encoder0/conv_1/kernel*
dtype0
�
:main_graph_0_encoder0/conv_1/kernel/Adam/Initializer/zerosFillJmain_graph_0_encoder0/conv_1/kernel/Adam/Initializer/zeros/shape_as_tensor@main_graph_0_encoder0/conv_1/kernel/Adam/Initializer/zeros/Const*
T0*

index_type0*6
_class,
*(loc:@main_graph_0_encoder0/conv_1/kernel
�
(main_graph_0_encoder0/conv_1/kernel/Adam
VariableV2*
shape:  *
shared_name *6
_class,
*(loc:@main_graph_0_encoder0/conv_1/kernel*
dtype0*
	container 
�
/main_graph_0_encoder0/conv_1/kernel/Adam/AssignAssign(main_graph_0_encoder0/conv_1/kernel/Adam:main_graph_0_encoder0/conv_1/kernel/Adam/Initializer/zeros*
use_locking(*
T0*6
_class,
*(loc:@main_graph_0_encoder0/conv_1/kernel*
validate_shape(
�
-main_graph_0_encoder0/conv_1/kernel/Adam/readIdentity(main_graph_0_encoder0/conv_1/kernel/Adam*
T0*6
_class,
*(loc:@main_graph_0_encoder0/conv_1/kernel
�
Lmain_graph_0_encoder0/conv_1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*%
valueB"              *6
_class,
*(loc:@main_graph_0_encoder0/conv_1/kernel*
dtype0
�
Bmain_graph_0_encoder0/conv_1/kernel/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *6
_class,
*(loc:@main_graph_0_encoder0/conv_1/kernel*
dtype0
�
<main_graph_0_encoder0/conv_1/kernel/Adam_1/Initializer/zerosFillLmain_graph_0_encoder0/conv_1/kernel/Adam_1/Initializer/zeros/shape_as_tensorBmain_graph_0_encoder0/conv_1/kernel/Adam_1/Initializer/zeros/Const*
T0*

index_type0*6
_class,
*(loc:@main_graph_0_encoder0/conv_1/kernel
�
*main_graph_0_encoder0/conv_1/kernel/Adam_1
VariableV2*
shape:  *
shared_name *6
_class,
*(loc:@main_graph_0_encoder0/conv_1/kernel*
dtype0*
	container 
�
1main_graph_0_encoder0/conv_1/kernel/Adam_1/AssignAssign*main_graph_0_encoder0/conv_1/kernel/Adam_1<main_graph_0_encoder0/conv_1/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*6
_class,
*(loc:@main_graph_0_encoder0/conv_1/kernel*
validate_shape(
�
/main_graph_0_encoder0/conv_1/kernel/Adam_1/readIdentity*main_graph_0_encoder0/conv_1/kernel/Adam_1*
T0*6
_class,
*(loc:@main_graph_0_encoder0/conv_1/kernel
�
8main_graph_0_encoder0/conv_1/bias/Adam/Initializer/zerosConst*
valueB *    *4
_class*
(&loc:@main_graph_0_encoder0/conv_1/bias*
dtype0
�
&main_graph_0_encoder0/conv_1/bias/Adam
VariableV2*
shape: *
shared_name *4
_class*
(&loc:@main_graph_0_encoder0/conv_1/bias*
dtype0*
	container 
�
-main_graph_0_encoder0/conv_1/bias/Adam/AssignAssign&main_graph_0_encoder0/conv_1/bias/Adam8main_graph_0_encoder0/conv_1/bias/Adam/Initializer/zeros*
use_locking(*
T0*4
_class*
(&loc:@main_graph_0_encoder0/conv_1/bias*
validate_shape(
�
+main_graph_0_encoder0/conv_1/bias/Adam/readIdentity&main_graph_0_encoder0/conv_1/bias/Adam*
T0*4
_class*
(&loc:@main_graph_0_encoder0/conv_1/bias
�
:main_graph_0_encoder0/conv_1/bias/Adam_1/Initializer/zerosConst*
valueB *    *4
_class*
(&loc:@main_graph_0_encoder0/conv_1/bias*
dtype0
�
(main_graph_0_encoder0/conv_1/bias/Adam_1
VariableV2*
shape: *
shared_name *4
_class*
(&loc:@main_graph_0_encoder0/conv_1/bias*
dtype0*
	container 
�
/main_graph_0_encoder0/conv_1/bias/Adam_1/AssignAssign(main_graph_0_encoder0/conv_1/bias/Adam_1:main_graph_0_encoder0/conv_1/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*4
_class*
(&loc:@main_graph_0_encoder0/conv_1/bias*
validate_shape(
�
-main_graph_0_encoder0/conv_1/bias/Adam_1/readIdentity(main_graph_0_encoder0/conv_1/bias/Adam_1*
T0*4
_class*
(&loc:@main_graph_0_encoder0/conv_1/bias
�
pmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
valueB" 
      *\
_classR
PNloc:@main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/kernel*
dtype0
�
fmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/kernel/Adam/Initializer/zeros/ConstConst*
valueB
 *    *\
_classR
PNloc:@main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/kernel*
dtype0
�
`main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/kernel/Adam/Initializer/zerosFillpmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/kernel/Adam/Initializer/zeros/shape_as_tensorfmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/kernel/Adam/Initializer/zeros/Const*
T0*

index_type0*\
_classR
PNloc:@main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/kernel
�
Nmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/kernel/Adam
VariableV2*
shape:	� *
shared_name *\
_classR
PNloc:@main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/kernel*
dtype0*
	container 
�
Umain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/kernel/Adam/AssignAssignNmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/kernel/Adam`main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/kernel/Adam/Initializer/zeros*
use_locking(*
T0*\
_classR
PNloc:@main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/kernel*
validate_shape(
�
Smain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/kernel/Adam/readIdentityNmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/kernel/Adam*
T0*\
_classR
PNloc:@main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/kernel
�
rmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB" 
      *\
_classR
PNloc:@main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/kernel*
dtype0
�
hmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/kernel/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *\
_classR
PNloc:@main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/kernel*
dtype0
�
bmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/kernel/Adam_1/Initializer/zerosFillrmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/kernel/Adam_1/Initializer/zeros/shape_as_tensorhmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/kernel/Adam_1/Initializer/zeros/Const*
T0*

index_type0*\
_classR
PNloc:@main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/kernel
�
Pmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/kernel/Adam_1
VariableV2*
shape:	� *
shared_name *\
_classR
PNloc:@main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/kernel*
dtype0*
	container 
�
Wmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/kernel/Adam_1/AssignAssignPmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/kernel/Adam_1bmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*\
_classR
PNloc:@main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/kernel*
validate_shape(
�
Umain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/kernel/Adam_1/readIdentityPmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/kernel/Adam_1*
T0*\
_classR
PNloc:@main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/kernel
�
^main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/bias/Adam/Initializer/zerosConst*
valueB *    *Z
_classP
NLloc:@main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/bias*
dtype0
�
Lmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/bias/Adam
VariableV2*
shape: *
shared_name *Z
_classP
NLloc:@main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/bias*
dtype0*
	container 
�
Smain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/bias/Adam/AssignAssignLmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/bias/Adam^main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/bias/Adam/Initializer/zeros*
use_locking(*
T0*Z
_classP
NLloc:@main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/bias*
validate_shape(
�
Qmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/bias/Adam/readIdentityLmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/bias/Adam*
T0*Z
_classP
NLloc:@main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/bias
�
`main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/bias/Adam_1/Initializer/zerosConst*
valueB *    *Z
_classP
NLloc:@main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/bias*
dtype0
�
Nmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/bias/Adam_1
VariableV2*
shape: *
shared_name *Z
_classP
NLloc:@main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/bias*
dtype0*
	container 
�
Umain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/bias/Adam_1/AssignAssignNmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/bias/Adam_1`main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*Z
_classP
NLloc:@main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/bias*
validate_shape(
�
Smain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/bias/Adam_1/readIdentityNmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/bias/Adam_1*
T0*Z
_classP
NLloc:@main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/bias
�
pmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
valueB"        *\
_classR
PNloc:@main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/kernel*
dtype0
�
fmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/kernel/Adam/Initializer/zeros/ConstConst*
valueB
 *    *\
_classR
PNloc:@main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/kernel*
dtype0
�
`main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/kernel/Adam/Initializer/zerosFillpmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/kernel/Adam/Initializer/zeros/shape_as_tensorfmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/kernel/Adam/Initializer/zeros/Const*
T0*

index_type0*\
_classR
PNloc:@main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/kernel
�
Nmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/kernel/Adam
VariableV2*
shape
:  *
shared_name *\
_classR
PNloc:@main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/kernel*
dtype0*
	container 
�
Umain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/kernel/Adam/AssignAssignNmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/kernel/Adam`main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/kernel/Adam/Initializer/zeros*
use_locking(*
T0*\
_classR
PNloc:@main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/kernel*
validate_shape(
�
Smain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/kernel/Adam/readIdentityNmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/kernel/Adam*
T0*\
_classR
PNloc:@main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/kernel
�
rmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB"        *\
_classR
PNloc:@main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/kernel*
dtype0
�
hmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/kernel/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *\
_classR
PNloc:@main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/kernel*
dtype0
�
bmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/kernel/Adam_1/Initializer/zerosFillrmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/kernel/Adam_1/Initializer/zeros/shape_as_tensorhmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/kernel/Adam_1/Initializer/zeros/Const*
T0*

index_type0*\
_classR
PNloc:@main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/kernel
�
Pmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/kernel/Adam_1
VariableV2*
shape
:  *
shared_name *\
_classR
PNloc:@main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/kernel*
dtype0*
	container 
�
Wmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/kernel/Adam_1/AssignAssignPmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/kernel/Adam_1bmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*\
_classR
PNloc:@main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/kernel*
validate_shape(
�
Umain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/kernel/Adam_1/readIdentityPmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/kernel/Adam_1*
T0*\
_classR
PNloc:@main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/kernel
�
^main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/bias/Adam/Initializer/zerosConst*
valueB *    *Z
_classP
NLloc:@main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/bias*
dtype0
�
Lmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/bias/Adam
VariableV2*
shape: *
shared_name *Z
_classP
NLloc:@main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/bias*
dtype0*
	container 
�
Smain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/bias/Adam/AssignAssignLmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/bias/Adam^main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/bias/Adam/Initializer/zeros*
use_locking(*
T0*Z
_classP
NLloc:@main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/bias*
validate_shape(
�
Qmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/bias/Adam/readIdentityLmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/bias/Adam*
T0*Z
_classP
NLloc:@main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/bias
�
`main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/bias/Adam_1/Initializer/zerosConst*
valueB *    *Z
_classP
NLloc:@main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/bias*
dtype0
�
Nmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/bias/Adam_1
VariableV2*
shape: *
shared_name *Z
_classP
NLloc:@main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/bias*
dtype0*
	container 
�
Umain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/bias/Adam_1/AssignAssignNmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/bias/Adam_1`main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*Z
_classP
NLloc:@main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/bias*
validate_shape(
�
Smain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/bias/Adam_1/readIdentityNmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/bias/Adam_1*
T0*Z
_classP
NLloc:@main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/bias
�
Flstm/rnn/basic_lstm_cell/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
valueB"�      *2
_class(
&$loc:@lstm/rnn/basic_lstm_cell/kernel*
dtype0
�
<lstm/rnn/basic_lstm_cell/kernel/Adam/Initializer/zeros/ConstConst*
valueB
 *    *2
_class(
&$loc:@lstm/rnn/basic_lstm_cell/kernel*
dtype0
�
6lstm/rnn/basic_lstm_cell/kernel/Adam/Initializer/zerosFillFlstm/rnn/basic_lstm_cell/kernel/Adam/Initializer/zeros/shape_as_tensor<lstm/rnn/basic_lstm_cell/kernel/Adam/Initializer/zeros/Const*
T0*

index_type0*2
_class(
&$loc:@lstm/rnn/basic_lstm_cell/kernel
�
$lstm/rnn/basic_lstm_cell/kernel/Adam
VariableV2*
shape:
��*
shared_name *2
_class(
&$loc:@lstm/rnn/basic_lstm_cell/kernel*
dtype0*
	container 
�
+lstm/rnn/basic_lstm_cell/kernel/Adam/AssignAssign$lstm/rnn/basic_lstm_cell/kernel/Adam6lstm/rnn/basic_lstm_cell/kernel/Adam/Initializer/zeros*
use_locking(*
T0*2
_class(
&$loc:@lstm/rnn/basic_lstm_cell/kernel*
validate_shape(
�
)lstm/rnn/basic_lstm_cell/kernel/Adam/readIdentity$lstm/rnn/basic_lstm_cell/kernel/Adam*
T0*2
_class(
&$loc:@lstm/rnn/basic_lstm_cell/kernel
�
Hlstm/rnn/basic_lstm_cell/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB"�      *2
_class(
&$loc:@lstm/rnn/basic_lstm_cell/kernel*
dtype0
�
>lstm/rnn/basic_lstm_cell/kernel/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *2
_class(
&$loc:@lstm/rnn/basic_lstm_cell/kernel*
dtype0
�
8lstm/rnn/basic_lstm_cell/kernel/Adam_1/Initializer/zerosFillHlstm/rnn/basic_lstm_cell/kernel/Adam_1/Initializer/zeros/shape_as_tensor>lstm/rnn/basic_lstm_cell/kernel/Adam_1/Initializer/zeros/Const*
T0*

index_type0*2
_class(
&$loc:@lstm/rnn/basic_lstm_cell/kernel
�
&lstm/rnn/basic_lstm_cell/kernel/Adam_1
VariableV2*
shape:
��*
shared_name *2
_class(
&$loc:@lstm/rnn/basic_lstm_cell/kernel*
dtype0*
	container 
�
-lstm/rnn/basic_lstm_cell/kernel/Adam_1/AssignAssign&lstm/rnn/basic_lstm_cell/kernel/Adam_18lstm/rnn/basic_lstm_cell/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*2
_class(
&$loc:@lstm/rnn/basic_lstm_cell/kernel*
validate_shape(
�
+lstm/rnn/basic_lstm_cell/kernel/Adam_1/readIdentity&lstm/rnn/basic_lstm_cell/kernel/Adam_1*
T0*2
_class(
&$loc:@lstm/rnn/basic_lstm_cell/kernel
�
4lstm/rnn/basic_lstm_cell/bias/Adam/Initializer/zerosConst*
valueB�*    *0
_class&
$"loc:@lstm/rnn/basic_lstm_cell/bias*
dtype0
�
"lstm/rnn/basic_lstm_cell/bias/Adam
VariableV2*
shape:�*
shared_name *0
_class&
$"loc:@lstm/rnn/basic_lstm_cell/bias*
dtype0*
	container 
�
)lstm/rnn/basic_lstm_cell/bias/Adam/AssignAssign"lstm/rnn/basic_lstm_cell/bias/Adam4lstm/rnn/basic_lstm_cell/bias/Adam/Initializer/zeros*
use_locking(*
T0*0
_class&
$"loc:@lstm/rnn/basic_lstm_cell/bias*
validate_shape(
�
'lstm/rnn/basic_lstm_cell/bias/Adam/readIdentity"lstm/rnn/basic_lstm_cell/bias/Adam*
T0*0
_class&
$"loc:@lstm/rnn/basic_lstm_cell/bias
�
6lstm/rnn/basic_lstm_cell/bias/Adam_1/Initializer/zerosConst*
valueB�*    *0
_class&
$"loc:@lstm/rnn/basic_lstm_cell/bias*
dtype0
�
$lstm/rnn/basic_lstm_cell/bias/Adam_1
VariableV2*
shape:�*
shared_name *0
_class&
$"loc:@lstm/rnn/basic_lstm_cell/bias*
dtype0*
	container 
�
+lstm/rnn/basic_lstm_cell/bias/Adam_1/AssignAssign$lstm/rnn/basic_lstm_cell/bias/Adam_16lstm/rnn/basic_lstm_cell/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*0
_class&
$"loc:@lstm/rnn/basic_lstm_cell/bias*
validate_shape(
�
)lstm/rnn/basic_lstm_cell/bias/Adam_1/readIdentity$lstm/rnn/basic_lstm_cell/bias/Adam_1*
T0*0
_class&
$"loc:@lstm/rnn/basic_lstm_cell/bias
z
#dense/kernel/Adam/Initializer/zerosConst*
valueB	�*    *
_class
loc:@dense/kernel*
dtype0
�
dense/kernel/Adam
VariableV2*
shape:	�*
shared_name *
_class
loc:@dense/kernel*
dtype0*
	container 
�
dense/kernel/Adam/AssignAssigndense/kernel/Adam#dense/kernel/Adam/Initializer/zeros*
use_locking(*
T0*
_class
loc:@dense/kernel*
validate_shape(
_
dense/kernel/Adam/readIdentitydense/kernel/Adam*
T0*
_class
loc:@dense/kernel
|
%dense/kernel/Adam_1/Initializer/zerosConst*
valueB	�*    *
_class
loc:@dense/kernel*
dtype0
�
dense/kernel/Adam_1
VariableV2*
shape:	�*
shared_name *
_class
loc:@dense/kernel*
dtype0*
	container 
�
dense/kernel/Adam_1/AssignAssigndense/kernel/Adam_1%dense/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*
_class
loc:@dense/kernel*
validate_shape(
c
dense/kernel/Adam_1/readIdentitydense/kernel/Adam_1*
T0*
_class
loc:@dense/kernel
�
-extrinsic_value/kernel/Adam/Initializer/zerosConst*
valueB	�*    *)
_class
loc:@extrinsic_value/kernel*
dtype0
�
extrinsic_value/kernel/Adam
VariableV2*
shape:	�*
shared_name *)
_class
loc:@extrinsic_value/kernel*
dtype0*
	container 
�
"extrinsic_value/kernel/Adam/AssignAssignextrinsic_value/kernel/Adam-extrinsic_value/kernel/Adam/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@extrinsic_value/kernel*
validate_shape(
}
 extrinsic_value/kernel/Adam/readIdentityextrinsic_value/kernel/Adam*
T0*)
_class
loc:@extrinsic_value/kernel
�
/extrinsic_value/kernel/Adam_1/Initializer/zerosConst*
valueB	�*    *)
_class
loc:@extrinsic_value/kernel*
dtype0
�
extrinsic_value/kernel/Adam_1
VariableV2*
shape:	�*
shared_name *)
_class
loc:@extrinsic_value/kernel*
dtype0*
	container 
�
$extrinsic_value/kernel/Adam_1/AssignAssignextrinsic_value/kernel/Adam_1/extrinsic_value/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@extrinsic_value/kernel*
validate_shape(
�
"extrinsic_value/kernel/Adam_1/readIdentityextrinsic_value/kernel/Adam_1*
T0*)
_class
loc:@extrinsic_value/kernel
�
+extrinsic_value/bias/Adam/Initializer/zerosConst*
valueB*    *'
_class
loc:@extrinsic_value/bias*
dtype0
�
extrinsic_value/bias/Adam
VariableV2*
shape:*
shared_name *'
_class
loc:@extrinsic_value/bias*
dtype0*
	container 
�
 extrinsic_value/bias/Adam/AssignAssignextrinsic_value/bias/Adam+extrinsic_value/bias/Adam/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@extrinsic_value/bias*
validate_shape(
w
extrinsic_value/bias/Adam/readIdentityextrinsic_value/bias/Adam*
T0*'
_class
loc:@extrinsic_value/bias
�
-extrinsic_value/bias/Adam_1/Initializer/zerosConst*
valueB*    *'
_class
loc:@extrinsic_value/bias*
dtype0
�
extrinsic_value/bias/Adam_1
VariableV2*
shape:*
shared_name *'
_class
loc:@extrinsic_value/bias*
dtype0*
	container 
�
"extrinsic_value/bias/Adam_1/AssignAssignextrinsic_value/bias/Adam_1-extrinsic_value/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@extrinsic_value/bias*
validate_shape(
{
 extrinsic_value/bias/Adam_1/readIdentityextrinsic_value/bias/Adam_1*
T0*'
_class
loc:@extrinsic_value/bias
7

Adam/beta1Const*
valueB
 *fff?*
dtype0
7

Adam/beta2Const*
valueB
 *w�?*
dtype0
9
Adam/epsilonConst*
valueB
 *w�+2*
dtype0
�
9Adam/update_main_graph_0_encoder0/conv_0/kernel/ApplyAdam	ApplyAdam#main_graph_0_encoder0/conv_0/kernel(main_graph_0_encoder0/conv_0/kernel/Adam*main_graph_0_encoder0/conv_0/kernel/Adam_1beta1_power/readbeta2_power/readPolynomialDecay
Adam/beta1
Adam/beta2Adam/epsilonOgradients_1/main_graph_0_encoder0/conv_0/Conv2D_grad/tuple/control_dependency_1*
use_locking( *
T0*6
_class,
*(loc:@main_graph_0_encoder0/conv_0/kernel*
use_nesterov( 
�
7Adam/update_main_graph_0_encoder0/conv_0/bias/ApplyAdam	ApplyAdam!main_graph_0_encoder0/conv_0/bias&main_graph_0_encoder0/conv_0/bias/Adam(main_graph_0_encoder0/conv_0/bias/Adam_1beta1_power/readbeta2_power/readPolynomialDecay
Adam/beta1
Adam/beta2Adam/epsilonPgradients_1/main_graph_0_encoder0/conv_0/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*4
_class*
(&loc:@main_graph_0_encoder0/conv_0/bias*
use_nesterov( 
�
9Adam/update_main_graph_0_encoder0/conv_1/kernel/ApplyAdam	ApplyAdam#main_graph_0_encoder0/conv_1/kernel(main_graph_0_encoder0/conv_1/kernel/Adam*main_graph_0_encoder0/conv_1/kernel/Adam_1beta1_power/readbeta2_power/readPolynomialDecay
Adam/beta1
Adam/beta2Adam/epsilonOgradients_1/main_graph_0_encoder0/conv_1/Conv2D_grad/tuple/control_dependency_1*
use_locking( *
T0*6
_class,
*(loc:@main_graph_0_encoder0/conv_1/kernel*
use_nesterov( 
�
7Adam/update_main_graph_0_encoder0/conv_1/bias/ApplyAdam	ApplyAdam!main_graph_0_encoder0/conv_1/bias&main_graph_0_encoder0/conv_1/bias/Adam(main_graph_0_encoder0/conv_1/bias/Adam_1beta1_power/readbeta2_power/readPolynomialDecay
Adam/beta1
Adam/beta2Adam/epsilonPgradients_1/main_graph_0_encoder0/conv_1/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*4
_class*
(&loc:@main_graph_0_encoder0/conv_1/bias*
use_nesterov( 
�
_Adam/update_main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/kernel/ApplyAdam	ApplyAdamImain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/kernelNmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/kernel/AdamPmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/kernel/Adam_1beta1_power/readbeta2_power/readPolynomialDecay
Adam/beta1
Adam/beta2Adam/epsilonugradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*\
_classR
PNloc:@main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/kernel*
use_nesterov( 
�
]Adam/update_main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/bias/ApplyAdam	ApplyAdamGmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/biasLmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/bias/AdamNmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/bias/Adam_1beta1_power/readbeta2_power/readPolynomialDecay
Adam/beta1
Adam/beta2Adam/epsilonvgradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*Z
_classP
NLloc:@main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/bias*
use_nesterov( 
�
_Adam/update_main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/kernel/ApplyAdam	ApplyAdamImain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/kernelNmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/kernel/AdamPmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/kernel/Adam_1beta1_power/readbeta2_power/readPolynomialDecay
Adam/beta1
Adam/beta2Adam/epsilonugradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*\
_classR
PNloc:@main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/kernel*
use_nesterov( 
�
]Adam/update_main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/bias/ApplyAdam	ApplyAdamGmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/biasLmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/bias/AdamNmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/bias/Adam_1beta1_power/readbeta2_power/readPolynomialDecay
Adam/beta1
Adam/beta2Adam/epsilonvgradients_1/main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*Z
_classP
NLloc:@main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/bias*
use_nesterov( 
�
5Adam/update_lstm/rnn/basic_lstm_cell/kernel/ApplyAdam	ApplyAdamlstm/rnn/basic_lstm_cell/kernel$lstm/rnn/basic_lstm_cell/kernel/Adam&lstm/rnn/basic_lstm_cell/kernel/Adam_1beta1_power/readbeta2_power/readPolynomialDecay
Adam/beta1
Adam/beta2Adam/epsilonDgradients_1/lstm/rnn/while/basic_lstm_cell/MatMul/Enter_grad/b_acc_3*
use_locking( *
T0*2
_class(
&$loc:@lstm/rnn/basic_lstm_cell/kernel*
use_nesterov( 
�
3Adam/update_lstm/rnn/basic_lstm_cell/bias/ApplyAdam	ApplyAdamlstm/rnn/basic_lstm_cell/bias"lstm/rnn/basic_lstm_cell/bias/Adam$lstm/rnn/basic_lstm_cell/bias/Adam_1beta1_power/readbeta2_power/readPolynomialDecay
Adam/beta1
Adam/beta2Adam/epsilonEgradients_1/lstm/rnn/while/basic_lstm_cell/BiasAdd/Enter_grad/b_acc_3*
use_locking( *
T0*0
_class&
$"loc:@lstm/rnn/basic_lstm_cell/bias*
use_nesterov( 
�
"Adam/update_dense/kernel/ApplyAdam	ApplyAdamdense/kerneldense/kernel/Adamdense/kernel/Adam_1beta1_power/readbeta2_power/readPolynomialDecay
Adam/beta1
Adam/beta2Adam/epsilon8gradients_1/dense/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*
_class
loc:@dense/kernel*
use_nesterov( 
�
,Adam/update_extrinsic_value/kernel/ApplyAdam	ApplyAdamextrinsic_value/kernelextrinsic_value/kernel/Adamextrinsic_value/kernel/Adam_1beta1_power/readbeta2_power/readPolynomialDecay
Adam/beta1
Adam/beta2Adam/epsilonBgradients_1/extrinsic_value/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*)
_class
loc:@extrinsic_value/kernel*
use_nesterov( 
�
*Adam/update_extrinsic_value/bias/ApplyAdam	ApplyAdamextrinsic_value/biasextrinsic_value/bias/Adamextrinsic_value/bias/Adam_1beta1_power/readbeta2_power/readPolynomialDecay
Adam/beta1
Adam/beta2Adam/epsilonCgradients_1/extrinsic_value/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*'
_class
loc:@extrinsic_value/bias*
use_nesterov( 
�
Adam/mulMulbeta1_power/read
Adam/beta1#^Adam/update_dense/kernel/ApplyAdam+^Adam/update_extrinsic_value/bias/ApplyAdam-^Adam/update_extrinsic_value/kernel/ApplyAdam4^Adam/update_lstm/rnn/basic_lstm_cell/bias/ApplyAdam6^Adam/update_lstm/rnn/basic_lstm_cell/kernel/ApplyAdam8^Adam/update_main_graph_0_encoder0/conv_0/bias/ApplyAdam:^Adam/update_main_graph_0_encoder0/conv_0/kernel/ApplyAdam8^Adam/update_main_graph_0_encoder0/conv_1/bias/ApplyAdam:^Adam/update_main_graph_0_encoder0/conv_1/kernel/ApplyAdam^^Adam/update_main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/bias/ApplyAdam`^Adam/update_main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/kernel/ApplyAdam^^Adam/update_main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/bias/ApplyAdam`^Adam/update_main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/kernel/ApplyAdam*
T0*
_class
loc:@dense/kernel

Adam/AssignAssignbeta1_powerAdam/mul*
use_locking( *
T0*
_class
loc:@dense/kernel*
validate_shape(
�

Adam/mul_1Mulbeta2_power/read
Adam/beta2#^Adam/update_dense/kernel/ApplyAdam+^Adam/update_extrinsic_value/bias/ApplyAdam-^Adam/update_extrinsic_value/kernel/ApplyAdam4^Adam/update_lstm/rnn/basic_lstm_cell/bias/ApplyAdam6^Adam/update_lstm/rnn/basic_lstm_cell/kernel/ApplyAdam8^Adam/update_main_graph_0_encoder0/conv_0/bias/ApplyAdam:^Adam/update_main_graph_0_encoder0/conv_0/kernel/ApplyAdam8^Adam/update_main_graph_0_encoder0/conv_1/bias/ApplyAdam:^Adam/update_main_graph_0_encoder0/conv_1/kernel/ApplyAdam^^Adam/update_main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/bias/ApplyAdam`^Adam/update_main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/kernel/ApplyAdam^^Adam/update_main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/bias/ApplyAdam`^Adam/update_main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/kernel/ApplyAdam*
T0*
_class
loc:@dense/kernel
�
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
use_locking( *
T0*
_class
loc:@dense/kernel*
validate_shape(
�
AdamNoOp^Adam/Assign^Adam/Assign_1#^Adam/update_dense/kernel/ApplyAdam+^Adam/update_extrinsic_value/bias/ApplyAdam-^Adam/update_extrinsic_value/kernel/ApplyAdam4^Adam/update_lstm/rnn/basic_lstm_cell/bias/ApplyAdam6^Adam/update_lstm/rnn/basic_lstm_cell/kernel/ApplyAdam8^Adam/update_main_graph_0_encoder0/conv_0/bias/ApplyAdam:^Adam/update_main_graph_0_encoder0/conv_0/kernel/ApplyAdam8^Adam/update_main_graph_0_encoder0/conv_1/bias/ApplyAdam:^Adam/update_main_graph_0_encoder0/conv_1/kernel/ApplyAdam^^Adam/update_main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/bias/ApplyAdam`^Adam/update_main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/kernel/ApplyAdam^^Adam/update_main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/bias/ApplyAdam`^Adam/update_main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/kernel/ApplyAdam
A
save/filename/inputConst*
valueB Bmodel*
dtype0
V
save/filenamePlaceholderWithDefaultsave/filename/input*
shape: *
dtype0
M

save/ConstPlaceholderWithDefaultsave/filename*
shape: *
dtype0
�
save/SaveV2/tensor_namesConst*�
value�B�.Baction_output_shapeBbeta1_powerBbeta2_powerBdense/kernelBdense/kernel/AdamBdense/kernel/Adam_1Bextrinsic_value/biasBextrinsic_value/bias/AdamBextrinsic_value/bias/Adam_1Bextrinsic_value/kernelBextrinsic_value/kernel/AdamBextrinsic_value/kernel/Adam_1Bglobal_stepBis_continuous_controlBlstm/rnn/basic_lstm_cell/biasB"lstm/rnn/basic_lstm_cell/bias/AdamB$lstm/rnn/basic_lstm_cell/bias/Adam_1Blstm/rnn/basic_lstm_cell/kernelB$lstm/rnn/basic_lstm_cell/kernel/AdamB&lstm/rnn/basic_lstm_cell/kernel/Adam_1B!main_graph_0_encoder0/conv_0/biasB&main_graph_0_encoder0/conv_0/bias/AdamB(main_graph_0_encoder0/conv_0/bias/Adam_1B#main_graph_0_encoder0/conv_0/kernelB(main_graph_0_encoder0/conv_0/kernel/AdamB*main_graph_0_encoder0/conv_0/kernel/Adam_1B!main_graph_0_encoder0/conv_1/biasB&main_graph_0_encoder0/conv_1/bias/AdamB(main_graph_0_encoder0/conv_1/bias/Adam_1B#main_graph_0_encoder0/conv_1/kernelB(main_graph_0_encoder0/conv_1/kernel/AdamB*main_graph_0_encoder0/conv_1/kernel/Adam_1BGmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/biasBLmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/bias/AdamBNmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/bias/Adam_1BImain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/kernelBNmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/kernel/AdamBPmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/kernel/Adam_1BGmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/biasBLmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/bias/AdamBNmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/bias/Adam_1BImain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/kernelBNmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/kernel/AdamBPmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/kernel/Adam_1Bmemory_sizeBversion_number*
dtype0
�
save/SaveV2/shape_and_slicesConst*o
valuefBd.B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
�
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesaction_output_shapebeta1_powerbeta2_powerdense/kerneldense/kernel/Adamdense/kernel/Adam_1extrinsic_value/biasextrinsic_value/bias/Adamextrinsic_value/bias/Adam_1extrinsic_value/kernelextrinsic_value/kernel/Adamextrinsic_value/kernel/Adam_1global_stepis_continuous_controllstm/rnn/basic_lstm_cell/bias"lstm/rnn/basic_lstm_cell/bias/Adam$lstm/rnn/basic_lstm_cell/bias/Adam_1lstm/rnn/basic_lstm_cell/kernel$lstm/rnn/basic_lstm_cell/kernel/Adam&lstm/rnn/basic_lstm_cell/kernel/Adam_1!main_graph_0_encoder0/conv_0/bias&main_graph_0_encoder0/conv_0/bias/Adam(main_graph_0_encoder0/conv_0/bias/Adam_1#main_graph_0_encoder0/conv_0/kernel(main_graph_0_encoder0/conv_0/kernel/Adam*main_graph_0_encoder0/conv_0/kernel/Adam_1!main_graph_0_encoder0/conv_1/bias&main_graph_0_encoder0/conv_1/bias/Adam(main_graph_0_encoder0/conv_1/bias/Adam_1#main_graph_0_encoder0/conv_1/kernel(main_graph_0_encoder0/conv_1/kernel/Adam*main_graph_0_encoder0/conv_1/kernel/Adam_1Gmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/biasLmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/bias/AdamNmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/bias/Adam_1Imain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/kernelNmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/kernel/AdamPmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/kernel/Adam_1Gmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/biasLmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/bias/AdamNmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/bias/Adam_1Imain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/kernelNmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/kernel/AdamPmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/kernel/Adam_1memory_sizeversion_number*<
dtypes2
02.
e
save/control_dependencyIdentity
save/Const^save/SaveV2*
T0*
_class
loc:@save/Const
�
save/RestoreV2/tensor_namesConst"/device:CPU:0*�
value�B�.Baction_output_shapeBbeta1_powerBbeta2_powerBdense/kernelBdense/kernel/AdamBdense/kernel/Adam_1Bextrinsic_value/biasBextrinsic_value/bias/AdamBextrinsic_value/bias/Adam_1Bextrinsic_value/kernelBextrinsic_value/kernel/AdamBextrinsic_value/kernel/Adam_1Bglobal_stepBis_continuous_controlBlstm/rnn/basic_lstm_cell/biasB"lstm/rnn/basic_lstm_cell/bias/AdamB$lstm/rnn/basic_lstm_cell/bias/Adam_1Blstm/rnn/basic_lstm_cell/kernelB$lstm/rnn/basic_lstm_cell/kernel/AdamB&lstm/rnn/basic_lstm_cell/kernel/Adam_1B!main_graph_0_encoder0/conv_0/biasB&main_graph_0_encoder0/conv_0/bias/AdamB(main_graph_0_encoder0/conv_0/bias/Adam_1B#main_graph_0_encoder0/conv_0/kernelB(main_graph_0_encoder0/conv_0/kernel/AdamB*main_graph_0_encoder0/conv_0/kernel/Adam_1B!main_graph_0_encoder0/conv_1/biasB&main_graph_0_encoder0/conv_1/bias/AdamB(main_graph_0_encoder0/conv_1/bias/Adam_1B#main_graph_0_encoder0/conv_1/kernelB(main_graph_0_encoder0/conv_1/kernel/AdamB*main_graph_0_encoder0/conv_1/kernel/Adam_1BGmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/biasBLmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/bias/AdamBNmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/bias/Adam_1BImain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/kernelBNmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/kernel/AdamBPmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/kernel/Adam_1BGmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/biasBLmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/bias/AdamBNmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/bias/Adam_1BImain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/kernelBNmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/kernel/AdamBPmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/kernel/Adam_1Bmemory_sizeBversion_number*
dtype0
�
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*o
valuefBd.B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
�
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*<
dtypes2
02.
�
save/AssignAssignaction_output_shapesave/RestoreV2*
use_locking(*
T0*&
_class
loc:@action_output_shape*
validate_shape(
�
save/Assign_1Assignbeta1_powersave/RestoreV2:1*
use_locking(*
T0*
_class
loc:@dense/kernel*
validate_shape(
�
save/Assign_2Assignbeta2_powersave/RestoreV2:2*
use_locking(*
T0*
_class
loc:@dense/kernel*
validate_shape(
�
save/Assign_3Assigndense/kernelsave/RestoreV2:3*
use_locking(*
T0*
_class
loc:@dense/kernel*
validate_shape(
�
save/Assign_4Assigndense/kernel/Adamsave/RestoreV2:4*
use_locking(*
T0*
_class
loc:@dense/kernel*
validate_shape(
�
save/Assign_5Assigndense/kernel/Adam_1save/RestoreV2:5*
use_locking(*
T0*
_class
loc:@dense/kernel*
validate_shape(
�
save/Assign_6Assignextrinsic_value/biassave/RestoreV2:6*
use_locking(*
T0*'
_class
loc:@extrinsic_value/bias*
validate_shape(
�
save/Assign_7Assignextrinsic_value/bias/Adamsave/RestoreV2:7*
use_locking(*
T0*'
_class
loc:@extrinsic_value/bias*
validate_shape(
�
save/Assign_8Assignextrinsic_value/bias/Adam_1save/RestoreV2:8*
use_locking(*
T0*'
_class
loc:@extrinsic_value/bias*
validate_shape(
�
save/Assign_9Assignextrinsic_value/kernelsave/RestoreV2:9*
use_locking(*
T0*)
_class
loc:@extrinsic_value/kernel*
validate_shape(
�
save/Assign_10Assignextrinsic_value/kernel/Adamsave/RestoreV2:10*
use_locking(*
T0*)
_class
loc:@extrinsic_value/kernel*
validate_shape(
�
save/Assign_11Assignextrinsic_value/kernel/Adam_1save/RestoreV2:11*
use_locking(*
T0*)
_class
loc:@extrinsic_value/kernel*
validate_shape(
�
save/Assign_12Assignglobal_stepsave/RestoreV2:12*
use_locking(*
T0*
_class
loc:@global_step*
validate_shape(
�
save/Assign_13Assignis_continuous_controlsave/RestoreV2:13*
use_locking(*
T0*(
_class
loc:@is_continuous_control*
validate_shape(
�
save/Assign_14Assignlstm/rnn/basic_lstm_cell/biassave/RestoreV2:14*
use_locking(*
T0*0
_class&
$"loc:@lstm/rnn/basic_lstm_cell/bias*
validate_shape(
�
save/Assign_15Assign"lstm/rnn/basic_lstm_cell/bias/Adamsave/RestoreV2:15*
use_locking(*
T0*0
_class&
$"loc:@lstm/rnn/basic_lstm_cell/bias*
validate_shape(
�
save/Assign_16Assign$lstm/rnn/basic_lstm_cell/bias/Adam_1save/RestoreV2:16*
use_locking(*
T0*0
_class&
$"loc:@lstm/rnn/basic_lstm_cell/bias*
validate_shape(
�
save/Assign_17Assignlstm/rnn/basic_lstm_cell/kernelsave/RestoreV2:17*
use_locking(*
T0*2
_class(
&$loc:@lstm/rnn/basic_lstm_cell/kernel*
validate_shape(
�
save/Assign_18Assign$lstm/rnn/basic_lstm_cell/kernel/Adamsave/RestoreV2:18*
use_locking(*
T0*2
_class(
&$loc:@lstm/rnn/basic_lstm_cell/kernel*
validate_shape(
�
save/Assign_19Assign&lstm/rnn/basic_lstm_cell/kernel/Adam_1save/RestoreV2:19*
use_locking(*
T0*2
_class(
&$loc:@lstm/rnn/basic_lstm_cell/kernel*
validate_shape(
�
save/Assign_20Assign!main_graph_0_encoder0/conv_0/biassave/RestoreV2:20*
use_locking(*
T0*4
_class*
(&loc:@main_graph_0_encoder0/conv_0/bias*
validate_shape(
�
save/Assign_21Assign&main_graph_0_encoder0/conv_0/bias/Adamsave/RestoreV2:21*
use_locking(*
T0*4
_class*
(&loc:@main_graph_0_encoder0/conv_0/bias*
validate_shape(
�
save/Assign_22Assign(main_graph_0_encoder0/conv_0/bias/Adam_1save/RestoreV2:22*
use_locking(*
T0*4
_class*
(&loc:@main_graph_0_encoder0/conv_0/bias*
validate_shape(
�
save/Assign_23Assign#main_graph_0_encoder0/conv_0/kernelsave/RestoreV2:23*
use_locking(*
T0*6
_class,
*(loc:@main_graph_0_encoder0/conv_0/kernel*
validate_shape(
�
save/Assign_24Assign(main_graph_0_encoder0/conv_0/kernel/Adamsave/RestoreV2:24*
use_locking(*
T0*6
_class,
*(loc:@main_graph_0_encoder0/conv_0/kernel*
validate_shape(
�
save/Assign_25Assign*main_graph_0_encoder0/conv_0/kernel/Adam_1save/RestoreV2:25*
use_locking(*
T0*6
_class,
*(loc:@main_graph_0_encoder0/conv_0/kernel*
validate_shape(
�
save/Assign_26Assign!main_graph_0_encoder0/conv_1/biassave/RestoreV2:26*
use_locking(*
T0*4
_class*
(&loc:@main_graph_0_encoder0/conv_1/bias*
validate_shape(
�
save/Assign_27Assign&main_graph_0_encoder0/conv_1/bias/Adamsave/RestoreV2:27*
use_locking(*
T0*4
_class*
(&loc:@main_graph_0_encoder0/conv_1/bias*
validate_shape(
�
save/Assign_28Assign(main_graph_0_encoder0/conv_1/bias/Adam_1save/RestoreV2:28*
use_locking(*
T0*4
_class*
(&loc:@main_graph_0_encoder0/conv_1/bias*
validate_shape(
�
save/Assign_29Assign#main_graph_0_encoder0/conv_1/kernelsave/RestoreV2:29*
use_locking(*
T0*6
_class,
*(loc:@main_graph_0_encoder0/conv_1/kernel*
validate_shape(
�
save/Assign_30Assign(main_graph_0_encoder0/conv_1/kernel/Adamsave/RestoreV2:30*
use_locking(*
T0*6
_class,
*(loc:@main_graph_0_encoder0/conv_1/kernel*
validate_shape(
�
save/Assign_31Assign*main_graph_0_encoder0/conv_1/kernel/Adam_1save/RestoreV2:31*
use_locking(*
T0*6
_class,
*(loc:@main_graph_0_encoder0/conv_1/kernel*
validate_shape(
�
save/Assign_32AssignGmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/biassave/RestoreV2:32*
use_locking(*
T0*Z
_classP
NLloc:@main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/bias*
validate_shape(
�
save/Assign_33AssignLmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/bias/Adamsave/RestoreV2:33*
use_locking(*
T0*Z
_classP
NLloc:@main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/bias*
validate_shape(
�
save/Assign_34AssignNmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/bias/Adam_1save/RestoreV2:34*
use_locking(*
T0*Z
_classP
NLloc:@main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/bias*
validate_shape(
�
save/Assign_35AssignImain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/kernelsave/RestoreV2:35*
use_locking(*
T0*\
_classR
PNloc:@main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/kernel*
validate_shape(
�
save/Assign_36AssignNmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/kernel/Adamsave/RestoreV2:36*
use_locking(*
T0*\
_classR
PNloc:@main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/kernel*
validate_shape(
�
save/Assign_37AssignPmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/kernel/Adam_1save/RestoreV2:37*
use_locking(*
T0*\
_classR
PNloc:@main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/kernel*
validate_shape(
�
save/Assign_38AssignGmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/biassave/RestoreV2:38*
use_locking(*
T0*Z
_classP
NLloc:@main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/bias*
validate_shape(
�
save/Assign_39AssignLmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/bias/Adamsave/RestoreV2:39*
use_locking(*
T0*Z
_classP
NLloc:@main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/bias*
validate_shape(
�
save/Assign_40AssignNmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/bias/Adam_1save/RestoreV2:40*
use_locking(*
T0*Z
_classP
NLloc:@main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/bias*
validate_shape(
�
save/Assign_41AssignImain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/kernelsave/RestoreV2:41*
use_locking(*
T0*\
_classR
PNloc:@main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/kernel*
validate_shape(
�
save/Assign_42AssignNmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/kernel/Adamsave/RestoreV2:42*
use_locking(*
T0*\
_classR
PNloc:@main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/kernel*
validate_shape(
�
save/Assign_43AssignPmain_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/kernel/Adam_1save/RestoreV2:43*
use_locking(*
T0*\
_classR
PNloc:@main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/kernel*
validate_shape(
�
save/Assign_44Assignmemory_sizesave/RestoreV2:44*
use_locking(*
T0*
_class
loc:@memory_size*
validate_shape(
�
save/Assign_45Assignversion_numbersave/RestoreV2:45*
use_locking(*
T0*!
_class
loc:@version_number*
validate_shape(
�
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_26^save/Assign_27^save/Assign_28^save/Assign_29^save/Assign_3^save/Assign_30^save/Assign_31^save/Assign_32^save/Assign_33^save/Assign_34^save/Assign_35^save/Assign_36^save/Assign_37^save/Assign_38^save/Assign_39^save/Assign_4^save/Assign_40^save/Assign_41^save/Assign_42^save/Assign_43^save/Assign_44^save/Assign_45^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9
�
initNoOp^action_output_shape/Assign^beta1_power/Assign^beta2_power/Assign^dense/kernel/Adam/Assign^dense/kernel/Adam_1/Assign^dense/kernel/Assign!^extrinsic_value/bias/Adam/Assign#^extrinsic_value/bias/Adam_1/Assign^extrinsic_value/bias/Assign#^extrinsic_value/kernel/Adam/Assign%^extrinsic_value/kernel/Adam_1/Assign^extrinsic_value/kernel/Assign^global_step/Assign^is_continuous_control/Assign*^lstm/rnn/basic_lstm_cell/bias/Adam/Assign,^lstm/rnn/basic_lstm_cell/bias/Adam_1/Assign%^lstm/rnn/basic_lstm_cell/bias/Assign,^lstm/rnn/basic_lstm_cell/kernel/Adam/Assign.^lstm/rnn/basic_lstm_cell/kernel/Adam_1/Assign'^lstm/rnn/basic_lstm_cell/kernel/Assign.^main_graph_0_encoder0/conv_0/bias/Adam/Assign0^main_graph_0_encoder0/conv_0/bias/Adam_1/Assign)^main_graph_0_encoder0/conv_0/bias/Assign0^main_graph_0_encoder0/conv_0/kernel/Adam/Assign2^main_graph_0_encoder0/conv_0/kernel/Adam_1/Assign+^main_graph_0_encoder0/conv_0/kernel/Assign.^main_graph_0_encoder0/conv_1/bias/Adam/Assign0^main_graph_0_encoder0/conv_1/bias/Adam_1/Assign)^main_graph_0_encoder0/conv_1/bias/Assign0^main_graph_0_encoder0/conv_1/kernel/Adam/Assign2^main_graph_0_encoder0/conv_1/kernel/Adam_1/Assign+^main_graph_0_encoder0/conv_1/kernel/AssignT^main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/bias/Adam/AssignV^main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/bias/Adam_1/AssignO^main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/bias/AssignV^main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/kernel/Adam/AssignX^main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/kernel/Adam_1/AssignQ^main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_2/kernel/AssignT^main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/bias/Adam/AssignV^main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/bias/Adam_1/AssignO^main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/bias/AssignV^main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/kernel/Adam/AssignX^main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/kernel/Adam_1/AssignQ^main_graph_0_encoder0/flat_encoding/main_graph_0_encoder0/hidden_3/kernel/Assign^memory_size/Assign^version_number/Assign"�