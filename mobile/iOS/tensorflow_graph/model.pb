
2
	the_inputPlaceholder*
shape: *
dtype0
f
zero_padding1d_1/Pad/paddingsConst*
dtype0*1
value(B&"                       
_
zero_padding1d_1/PadPad	the_inputzero_padding1d_1/Pad/paddings*
T0*
	Tpaddings0
T
conv_1/random_uniform/shapeConst*!
valueB"   ”      *
dtype0
F
conv_1/random_uniform/minConst*
valueB
 *
»*
dtype0
F
conv_1/random_uniform/maxConst*
valueB
 *
;*
dtype0

#conv_1/random_uniform/RandomUniformRandomUniformconv_1/random_uniform/shape*
dtype0*
seed±’å)*
T0*
seed2ŹĪ
_
conv_1/random_uniform/subSubconv_1/random_uniform/maxconv_1/random_uniform/min*
T0
i
conv_1/random_uniform/mulMul#conv_1/random_uniform/RandomUniformconv_1/random_uniform/sub*
T0
[
conv_1/random_uniformAddconv_1/random_uniform/mulconv_1/random_uniform/min*
T0
g
conv_1/kernel
VariableV2*
	container *
shape:”*
dtype0*
shared_name 

conv_1/kernel/AssignAssignconv_1/kernelconv_1/random_uniform*
use_locking(*
validate_shape(*
T0* 
_class
loc:@conv_1/kernel
X
conv_1/kernel/readIdentityconv_1/kernel*
T0* 
_class
loc:@conv_1/kernel
>
conv_1/ConstConst*
valueB*    *
dtype0
\
conv_1/bias
VariableV2*
shared_name *
dtype0*
shape:*
	container 

conv_1/bias/AssignAssignconv_1/biasconv_1/Const*
use_locking(*
validate_shape(*
T0*
_class
loc:@conv_1/bias
R
conv_1/bias/readIdentityconv_1/bias*
T0*
_class
loc:@conv_1/bias
Q
conv_1/convolution/ShapeConst*
dtype0*!
valueB"   ”      
N
 conv_1/convolution/dilation_rateConst*
dtype0*
valueB:
K
!conv_1/convolution/ExpandDims/dimConst*
value	B :*
dtype0
y
conv_1/convolution/ExpandDims
ExpandDimszero_padding1d_1/Pad!conv_1/convolution/ExpandDims/dim*

Tdim0*
T0
M
#conv_1/convolution/ExpandDims_1/dimConst*
dtype0*
value	B : 
{
conv_1/convolution/ExpandDims_1
ExpandDimsconv_1/kernel/read#conv_1/convolution/ExpandDims_1/dim*

Tdim0*
T0
Ä
conv_1/convolution/Conv2DConv2Dconv_1/convolution/ExpandDimsconv_1/convolution/ExpandDims_1*
use_cudnn_on_gpu(*
data_formatNHWC*
strides
*
T0*
paddingVALID
`
conv_1/convolution/SqueezeSqueezeconv_1/convolution/Conv2D*
T0*
squeeze_dims

M
conv_1/Reshape/shapeConst*!
valueB"         *
dtype0
X
conv_1/ReshapeReshapeconv_1/bias/readconv_1/Reshape/shape*
T0*
Tshape0
F

conv_1/addAddconv_1/convolution/Squeezeconv_1/Reshape*
T0
(
conv_1/ReluRelu
conv_1/add*
T0
\
'time_distributed_1/random_uniform/shapeConst*
valueB"      *
dtype0
R
%time_distributed_1/random_uniform/minConst*
valueB
 *qÄ½*
dtype0
R
%time_distributed_1/random_uniform/maxConst*
dtype0*
valueB
 *qÄ=

/time_distributed_1/random_uniform/RandomUniformRandomUniform'time_distributed_1/random_uniform/shape*
seed2§šÜ*
dtype0*
T0*
seed±’å)

%time_distributed_1/random_uniform/subSub%time_distributed_1/random_uniform/max%time_distributed_1/random_uniform/min*
T0

%time_distributed_1/random_uniform/mulMul/time_distributed_1/random_uniform/RandomUniform%time_distributed_1/random_uniform/sub*
T0

!time_distributed_1/random_uniformAdd%time_distributed_1/random_uniform/mul%time_distributed_1/random_uniform/min*
T0
o
time_distributed_1/kernel
VariableV2*
shape:
*
shared_name *
dtype0*
	container 
Č
 time_distributed_1/kernel/AssignAssigntime_distributed_1/kernel!time_distributed_1/random_uniform*,
_class"
 loc:@time_distributed_1/kernel*
T0*
validate_shape(*
use_locking(
|
time_distributed_1/kernel/readIdentitytime_distributed_1/kernel*
T0*,
_class"
 loc:@time_distributed_1/kernel
J
time_distributed_1/ConstConst*
dtype0*
valueB*    
h
time_distributed_1/bias
VariableV2*
shared_name *
dtype0*
shape:*
	container 
¹
time_distributed_1/bias/AssignAssigntime_distributed_1/biastime_distributed_1/Const*
validate_shape(**
_class 
loc:@time_distributed_1/bias*
T0*
use_locking(
v
time_distributed_1/bias/readIdentitytime_distributed_1/bias*
T0**
_class 
loc:@time_distributed_1/bias
G
time_distributed_1/ShapeShapeconv_1/Relu*
out_type0*
T0
T
&time_distributed_1/strided_slice/stackConst*
valueB:*
dtype0
V
(time_distributed_1/strided_slice/stack_1Const*
valueB:*
dtype0
V
(time_distributed_1/strided_slice/stack_2Const*
dtype0*
valueB:
Ą
 time_distributed_1/strided_sliceStridedSlicetime_distributed_1/Shape&time_distributed_1/strided_slice/stack(time_distributed_1/strided_slice/stack_1(time_distributed_1/strided_slice/stack_2*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
T0*
Index0*
shrink_axis_mask
U
 time_distributed_1/Reshape/shapeConst*
valueB"’’’’   *
dtype0
k
time_distributed_1/ReshapeReshapeconv_1/Relu time_distributed_1/Reshape/shape*
T0*
Tshape0

time_distributed_1/MatMulMatMultime_distributed_1/Reshapetime_distributed_1/kernel/read*
transpose_b( *
transpose_a( *
T0
~
time_distributed_1/BiasAddBiasAddtime_distributed_1/MatMultime_distributed_1/bias/read*
T0*
data_formatNHWC
D
time_distributed_1/ReluRelutime_distributed_1/BiasAdd*
T0
W
$time_distributed_1/Reshape_1/shape/0Const*
valueB :
’’’’’’’’’*
dtype0
O
$time_distributed_1/Reshape_1/shape/2Const*
dtype0*
value
B :
¶
"time_distributed_1/Reshape_1/shapePack$time_distributed_1/Reshape_1/shape/0 time_distributed_1/strided_slice$time_distributed_1/Reshape_1/shape/2*
N*

axis *
T0
{
time_distributed_1/Reshape_1Reshapetime_distributed_1/Relu"time_distributed_1/Reshape_1/shape*
T0*
Tshape0
\
'time_distributed_2/random_uniform/shapeConst*
valueB"      *
dtype0
R
%time_distributed_2/random_uniform/minConst*
valueB
 *µ­×½*
dtype0
R
%time_distributed_2/random_uniform/maxConst*
dtype0*
valueB
 *µ­×=

/time_distributed_2/random_uniform/RandomUniformRandomUniform'time_distributed_2/random_uniform/shape*
dtype0*
seed±’å)*
T0*
seed2÷ź~

%time_distributed_2/random_uniform/subSub%time_distributed_2/random_uniform/max%time_distributed_2/random_uniform/min*
T0

%time_distributed_2/random_uniform/mulMul/time_distributed_2/random_uniform/RandomUniform%time_distributed_2/random_uniform/sub*
T0

!time_distributed_2/random_uniformAdd%time_distributed_2/random_uniform/mul%time_distributed_2/random_uniform/min*
T0
n
time_distributed_2/kernel
VariableV2*
shape:	*
shared_name *
dtype0*
	container 
Č
 time_distributed_2/kernel/AssignAssigntime_distributed_2/kernel!time_distributed_2/random_uniform*
use_locking(*
T0*,
_class"
 loc:@time_distributed_2/kernel*
validate_shape(
|
time_distributed_2/kernel/readIdentitytime_distributed_2/kernel*,
_class"
 loc:@time_distributed_2/kernel*
T0
I
time_distributed_2/ConstConst*
valueB*    *
dtype0
g
time_distributed_2/bias
VariableV2*
	container *
dtype0*
shared_name *
shape:
¹
time_distributed_2/bias/AssignAssigntime_distributed_2/biastime_distributed_2/Const*
validate_shape(**
_class 
loc:@time_distributed_2/bias*
T0*
use_locking(
v
time_distributed_2/bias/readIdentitytime_distributed_2/bias*
T0**
_class 
loc:@time_distributed_2/bias
X
time_distributed_2/ShapeShapetime_distributed_1/Reshape_1*
out_type0*
T0
T
&time_distributed_2/strided_slice/stackConst*
valueB:*
dtype0
V
(time_distributed_2/strided_slice/stack_1Const*
dtype0*
valueB:
V
(time_distributed_2/strided_slice/stack_2Const*
dtype0*
valueB:
Ą
 time_distributed_2/strided_sliceStridedSlicetime_distributed_2/Shape&time_distributed_2/strided_slice/stack(time_distributed_2/strided_slice/stack_1(time_distributed_2/strided_slice/stack_2*
end_mask *

begin_mask *
ellipsis_mask *
shrink_axis_mask*
new_axis_mask *
T0*
Index0
U
 time_distributed_2/Reshape/shapeConst*
valueB"’’’’   *
dtype0
|
time_distributed_2/ReshapeReshapetime_distributed_1/Reshape_1 time_distributed_2/Reshape/shape*
T0*
Tshape0

time_distributed_2/MatMulMatMultime_distributed_2/Reshapetime_distributed_2/kernel/read*
transpose_b( *
T0*
transpose_a( 
~
time_distributed_2/BiasAddBiasAddtime_distributed_2/MatMultime_distributed_2/bias/read*
T0*
data_formatNHWC
J
time_distributed_2/SoftmaxSoftmaxtime_distributed_2/BiasAdd*
T0
W
$time_distributed_2/Reshape_1/shape/0Const*
valueB :
’’’’’’’’’*
dtype0
N
$time_distributed_2/Reshape_1/shape/2Const*
value	B :*
dtype0
¶
"time_distributed_2/Reshape_1/shapePack$time_distributed_2/Reshape_1/shape/0 time_distributed_2/strided_slice$time_distributed_2/Reshape_1/shape/2*
T0*

axis *
N
~
time_distributed_2/Reshape_1Reshapetime_distributed_2/Softmax"time_distributed_2/Reshape_1/shape*
T0*
Tshape0
3

the_labelsPlaceholder*
dtype0*
shape: 
5
input_lengthPlaceholder*
shape: *
dtype0
5
label_lengthPlaceholder*
shape: *
dtype0
A
ctc/SqueezeSqueezelabel_length*
T0*
squeeze_dims
 
C
ctc/Squeeze_1Squeezeinput_length*
T0*
squeeze_dims
 
7
	ctc/ShapeShape
the_labels*
out_type0*
T0
E
ctc/strided_slice/stackConst*
dtype0*
valueB: 
G
ctc/strided_slice/stack_1Const*
valueB:*
dtype0
G
ctc/strided_slice/stack_2Const*
dtype0*
valueB:
õ
ctc/strided_sliceStridedSlice	ctc/Shapectc/strided_slice/stackctc/strided_slice/stack_1ctc/strided_slice/stack_2*
new_axis_mask *
shrink_axis_mask*
T0*
Index0*
end_mask *

begin_mask *
ellipsis_mask 
B
	ctc/stackPackctc/strided_slice*
N*

axis *
T0
G
ctc/strided_slice_1/stackConst*
dtype0*
valueB:
I
ctc/strided_slice_1/stack_1Const*
dtype0*
valueB:
I
ctc/strided_slice_1/stack_2Const*
valueB:*
dtype0
ż
ctc/strided_slice_1StridedSlice	ctc/Shapectc/strided_slice_1/stackctc/strided_slice_1/stack_1ctc/strided_slice_1/stack_2*
end_mask *

begin_mask *
ellipsis_mask *
shrink_axis_mask*
new_axis_mask *
T0*
Index0
F
ctc/stack_1Packctc/strided_slice_1*
T0*

axis *
N
G
ctc/strided_slice_2/stackConst*
valueB:*
dtype0
I
ctc/strided_slice_2/stack_1Const*
dtype0*
valueB:
I
ctc/strided_slice_2/stack_2Const*
dtype0*
valueB:
ż
ctc/strided_slice_2StridedSlice	ctc/Shapectc/strided_slice_2/stackctc/strided_slice_2/stack_1ctc/strided_slice_2/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask 
9
ctc/Fill/dims/0Const*
dtype0*
value	B :
Y
ctc/Fill/dimsPackctc/Fill/dims/0ctc/strided_slice_2*
T0*

axis *
N
8
ctc/Fill/valueConst*
value	B : *
dtype0
8
ctc/FillFillctc/Fill/dimsctc/Fill/value*
T0
2
ctc/CastCastctc/Fill*

DstT0
*

SrcT0
=
ctc/scan/ShapeShapectc/Squeeze*
out_type0*
T0
J
ctc/scan/strided_slice/stackConst*
valueB: *
dtype0
L
ctc/scan/strided_slice/stack_1Const*
valueB:*
dtype0
L
ctc/scan/strided_slice/stack_2Const*
dtype0*
valueB:

ctc/scan/strided_sliceStridedSlicectc/scan/Shapectc/scan/strided_slice/stackctc/scan/strided_slice/stack_1ctc/scan/strided_slice/stack_2*
T0*
Index0*
new_axis_mask *
shrink_axis_mask*

begin_mask *
ellipsis_mask *
end_mask 
¦
ctc/scan/TensorArrayTensorArrayV3ctc/scan/strided_slice*
dtype0*
tensor_array_name *
dynamic_size( *
clear_after_read(*
element_shape:
P
!ctc/scan/TensorArrayUnstack/ShapeShapectc/Squeeze*
T0*
out_type0
]
/ctc/scan/TensorArrayUnstack/strided_slice/stackConst*
valueB: *
dtype0
_
1ctc/scan/TensorArrayUnstack/strided_slice/stack_1Const*
dtype0*
valueB:
_
1ctc/scan/TensorArrayUnstack/strided_slice/stack_2Const*
dtype0*
valueB:
ķ
)ctc/scan/TensorArrayUnstack/strided_sliceStridedSlice!ctc/scan/TensorArrayUnstack/Shape/ctc/scan/TensorArrayUnstack/strided_slice/stack1ctc/scan/TensorArrayUnstack/strided_slice/stack_11ctc/scan/TensorArrayUnstack/strided_slice/stack_2*
end_mask *

begin_mask *
ellipsis_mask *
shrink_axis_mask*
new_axis_mask *
T0*
Index0
Q
'ctc/scan/TensorArrayUnstack/range/startConst*
dtype0*
value	B : 
Q
'ctc/scan/TensorArrayUnstack/range/deltaConst*
dtype0*
value	B :
³
!ctc/scan/TensorArrayUnstack/rangeRange'ctc/scan/TensorArrayUnstack/range/start)ctc/scan/TensorArrayUnstack/strided_slice'ctc/scan/TensorArrayUnstack/range/delta*

Tidx0
ė
Cctc/scan/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3ctc/scan/TensorArray!ctc/scan/TensorArrayUnstack/rangectc/Squeezectc/scan/TensorArray:1*
T0*'
_class
loc:@ctc/scan/TensorArray
8
ctc/scan/ConstConst*
value	B : *
dtype0
Ø
ctc/scan/TensorArray_1TensorArrayV3ctc/scan/strided_slice*
element_shape:*
dynamic_size( *
clear_after_read(*
tensor_array_name *
dtype0


ctc/scan/while/EnterEnterctc/scan/Const*.

frame_name ctc/scan/while/ctc/scan/while/*
parallel_iterations*
is_constant( *
T0

ctc/scan/while/Enter_1Enterctc/Cast*
is_constant( *.

frame_name ctc/scan/while/ctc/scan/while/*
T0
*
parallel_iterations
 
ctc/scan/while/Enter_2Enterctc/scan/TensorArray_1:1*
is_constant( *.

frame_name ctc/scan/while/ctc/scan/while/*
T0*
parallel_iterations
c
ctc/scan/while/MergeMergectc/scan/while/Enterctc/scan/while/NextIteration*
T0*
N
i
ctc/scan/while/Merge_1Mergectc/scan/while/Enter_1ctc/scan/while/NextIteration_1*
T0
*
N
i
ctc/scan/while/Merge_2Mergectc/scan/while/Enter_2ctc/scan/while/NextIteration_2*
T0*
N
”
ctc/scan/while/Less/EnterEnterctc/scan/strided_slice*
parallel_iterations*
T0*.

frame_name ctc/scan/while/ctc/scan/while/*
is_constant(
U
ctc/scan/while/LessLessctc/scan/while/Mergectc/scan/while/Less/Enter*
T0
8
ctc/scan/while/LoopCondLoopCondctc/scan/while/Less

ctc/scan/while/SwitchSwitchctc/scan/while/Mergectc/scan/while/LoopCond*
T0*'
_class
loc:@ctc/scan/while/Merge

ctc/scan/while/Switch_1Switchctc/scan/while/Merge_1ctc/scan/while/LoopCond*
T0
*)
_class
loc:@ctc/scan/while/Merge_1

ctc/scan/while/Switch_2Switchctc/scan/while/Merge_2ctc/scan/while/LoopCond*)
_class
loc:@ctc/scan/while/Merge_2*
T0
E
ctc/scan/while/IdentityIdentityctc/scan/while/Switch:1*
T0
I
ctc/scan/while/Identity_1Identityctc/scan/while/Switch_1:1*
T0

I
ctc/scan/while/Identity_2Identityctc/scan/while/Switch_2:1*
T0
Õ
&ctc/scan/while/TensorArrayReadV3/EnterEnterctc/scan/TensorArray*'
_class
loc:@ctc/scan/TensorArray*
is_constant(*.

frame_name ctc/scan/while/ctc/scan/while/*
T0*
parallel_iterations

(ctc/scan/while/TensorArrayReadV3/Enter_1EnterCctc/scan/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
T0*'
_class
loc:@ctc/scan/TensorArray*
is_constant(*
parallel_iterations*.

frame_name ctc/scan/while/ctc/scan/while/
Ö
 ctc/scan/while/TensorArrayReadV3TensorArrayReadV3&ctc/scan/while/TensorArrayReadV3/Enterctc/scan/while/Identity(ctc/scan/while/TensorArrayReadV3/Enter_1*
dtype0*'
_class
loc:@ctc/scan/TensorArray
j
"ctc/scan/while/strided_slice/stackConst^ctc/scan/while/Identity*
valueB:*
dtype0
l
$ctc/scan/while/strided_slice/stack_1Const^ctc/scan/while/Identity*
dtype0*
valueB:
l
$ctc/scan/while/strided_slice/stack_2Const^ctc/scan/while/Identity*
dtype0*
valueB:

"ctc/scan/while/strided_slice/EnterEnter	ctc/Shape*
T0*
is_constant(*
parallel_iterations*.

frame_name ctc/scan/while/ctc/scan/while/
ŗ
ctc/scan/while/strided_sliceStridedSlice"ctc/scan/while/strided_slice/Enter"ctc/scan/while/strided_slice/stack$ctc/scan/while/strided_slice/stack_1$ctc/scan/while/strided_slice/stack_2*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
T0*
Index0*
shrink_axis_mask
^
ctc/scan/while/range/startConst^ctc/scan/while/Identity*
dtype0*
value	B : 
^
ctc/scan/while/range/deltaConst^ctc/scan/while/Identity*
value	B :*
dtype0

ctc/scan/while/rangeRangectc/scan/while/range/startctc/scan/while/strided_slicectc/scan/while/range/delta*

Tidx0
a
ctc/scan/while/ExpandDims/dimConst^ctc/scan/while/Identity*
dtype0*
value	B : 
q
ctc/scan/while/ExpandDims
ExpandDimsctc/scan/while/rangectc/scan/while/ExpandDims/dim*
T0*

Tdim0

ctc/scan/while/Fill/EnterEnterctc/stack_1*
parallel_iterations*
T0*.

frame_name ctc/scan/while/ctc/scan/while/*
is_constant(
a
ctc/scan/while/FillFillctc/scan/while/Fill/Enter ctc/scan/while/TensorArrayReadV3*
T0
V
ctc/scan/while/Less_1Lessctc/scan/while/ExpandDimsctc/scan/while/Fill*
T0
ė
8ctc/scan/while/TensorArrayWrite/TensorArrayWriteV3/EnterEnterctc/scan/TensorArray_1*
T0*)
_class
loc:@ctc/scan/TensorArray_1*
is_constant(*
parallel_iterations*.

frame_name ctc/scan/while/ctc/scan/while/

2ctc/scan/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV38ctc/scan/while/TensorArrayWrite/TensorArrayWriteV3/Enterctc/scan/while/Identityctc/scan/while/Less_1ctc/scan/while/Identity_2*
T0
*)
_class
loc:@ctc/scan/TensorArray_1
X
ctc/scan/while/add/yConst^ctc/scan/while/Identity*
value	B :*
dtype0
Q
ctc/scan/while/addAddctc/scan/while/Identityctc/scan/while/add/y*
T0
J
ctc/scan/while/NextIterationNextIterationctc/scan/while/add*
T0
O
ctc/scan/while/NextIteration_1NextIterationctc/scan/while/Less_1*
T0

l
ctc/scan/while/NextIteration_2NextIteration2ctc/scan/while/TensorArrayWrite/TensorArrayWriteV3*
T0
;
ctc/scan/while/ExitExitctc/scan/while/Switch*
T0
?
ctc/scan/while/Exit_1Exitctc/scan/while/Switch_1*
T0

?
ctc/scan/while/Exit_2Exitctc/scan/while/Switch_2*
T0

+ctc/scan/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3ctc/scan/TensorArray_1ctc/scan/while/Exit_2*)
_class
loc:@ctc/scan/TensorArray_1
z
%ctc/scan/TensorArrayStack/range/startConst*)
_class
loc:@ctc/scan/TensorArray_1*
value	B : *
dtype0
z
%ctc/scan/TensorArrayStack/range/deltaConst*
dtype0*)
_class
loc:@ctc/scan/TensorArray_1*
value	B :
Ś
ctc/scan/TensorArrayStack/rangeRange%ctc/scan/TensorArrayStack/range/start+ctc/scan/TensorArrayStack/TensorArraySizeV3%ctc/scan/TensorArrayStack/range/delta*

Tidx0*)
_class
loc:@ctc/scan/TensorArray_1
ņ
-ctc/scan/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3ctc/scan/TensorArray_1ctc/scan/TensorArrayStack/rangectc/scan/while/Exit_2*)
_class
loc:@ctc/scan/TensorArray_1*$
element_shape:’’’’’’’’’*
dtype0

R
ctc/strided_slice_3/stackConst*
dtype0*!
valueB"            
T
ctc/strided_slice_3/stack_1Const*
dtype0*!
valueB"           
T
ctc/strided_slice_3/stack_2Const*!
valueB"         *
dtype0
”
ctc/strided_slice_3StridedSlice-ctc/scan/TensorArrayStack/TensorArrayGatherV3ctc/strided_slice_3/stackctc/strided_slice_3/stack_1ctc/strided_slice_3/stack_2*
new_axis_mask *
shrink_axis_mask*
T0
*
Index0*
end_mask*

begin_mask*
ellipsis_mask 
G
ctc/strided_slice_4/stackConst*
dtype0*
valueB:
I
ctc/strided_slice_4/stack_1Const*
dtype0*
valueB:
I
ctc/strided_slice_4/stack_2Const*
dtype0*
valueB:
ż
ctc/strided_slice_4StridedSlice	ctc/Shapectc/strided_slice_4/stackctc/strided_slice_4/stack_1ctc/strided_slice_4/stack_2*
T0*
Index0*
new_axis_mask *
shrink_axis_mask*

begin_mask *
ellipsis_mask *
end_mask 
9
ctc/range/startConst*
dtype0*
value	B : 
9
ctc/range/deltaConst*
dtype0*
value	B :
U
	ctc/rangeRangectc/range/startctc/strided_slice_4ctc/range/delta*

Tidx0
A
ctc/TileTile	ctc/range	ctc/stack*

Tmultiples0*
T0
B
ctc/ReshapeReshapectc/Tile	ctc/Shape*
Tshape0*
T0
E
ctc/boolean_mask/ShapeShapectc/Reshape*
T0*
out_type0
R
$ctc/boolean_mask/strided_slice/stackConst*
dtype0*
valueB: 
T
&ctc/boolean_mask/strided_slice/stack_1Const*
valueB:*
dtype0
T
&ctc/boolean_mask/strided_slice/stack_2Const*
valueB:*
dtype0
¶
ctc/boolean_mask/strided_sliceStridedSlicectc/boolean_mask/Shape$ctc/boolean_mask/strided_slice/stack&ctc/boolean_mask/strided_slice/stack_1&ctc/boolean_mask/strided_slice/stack_2*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask *
T0*
Index0*
shrink_axis_mask 
U
'ctc/boolean_mask/Prod/reduction_indicesConst*
valueB: *
dtype0

ctc/boolean_mask/ProdProdctc/boolean_mask/strided_slice'ctc/boolean_mask/Prod/reduction_indices*
T0*

Tidx0*
	keep_dims( 
G
ctc/boolean_mask/Shape_1Shapectc/Reshape*
T0*
out_type0
T
&ctc/boolean_mask/strided_slice_1/stackConst*
dtype0*
valueB:
V
(ctc/boolean_mask/strided_slice_1/stack_1Const*
valueB: *
dtype0
V
(ctc/boolean_mask/strided_slice_1/stack_2Const*
dtype0*
valueB:
Ą
 ctc/boolean_mask/strided_slice_1StridedSlicectc/boolean_mask/Shape_1&ctc/boolean_mask/strided_slice_1/stack(ctc/boolean_mask/strided_slice_1/stack_1(ctc/boolean_mask/strided_slice_1/stack_2*
end_mask*

begin_mask *
ellipsis_mask *
shrink_axis_mask *
new_axis_mask *
T0*
Index0
]
 ctc/boolean_mask/concat/values_0Packctc/boolean_mask/Prod*
N*
T0*

axis 
F
ctc/boolean_mask/concat/axisConst*
dtype0*
value	B : 
£
ctc/boolean_mask/concatConcatV2 ctc/boolean_mask/concat/values_0 ctc/boolean_mask/strided_slice_1ctc/boolean_mask/concat/axis*
N*

Tidx0*
T0
`
ctc/boolean_mask/ReshapeReshapectc/Reshapectc/boolean_mask/concat*
T0*
Tshape0
W
 ctc/boolean_mask/Reshape_1/shapeConst*
dtype0*
valueB:
’’’’’’’’’
s
ctc/boolean_mask/Reshape_1Reshapectc/strided_slice_3 ctc/boolean_mask/Reshape_1/shape*
T0
*
Tshape0
;
ctc/boolean_mask/WhereWherectc/boolean_mask/Reshape_1
[
ctc/boolean_mask/SqueezeSqueezectc/boolean_mask/Where*
T0	*
squeeze_dims


ctc/boolean_mask/GatherGatherctc/boolean_mask/Reshapectc/boolean_mask/Squeeze*
Tparams0*
validate_indices(*
Tindices0	
G
ctc/strided_slice_5/stackConst*
valueB: *
dtype0
I
ctc/strided_slice_5/stack_1Const*
dtype0*
valueB:
I
ctc/strided_slice_5/stack_2Const*
dtype0*
valueB:
ż
ctc/strided_slice_5StridedSlice	ctc/Shapectc/strided_slice_5/stackctc/strided_slice_5/stack_1ctc/strided_slice_5/stack_2*
end_mask *
new_axis_mask *

begin_mask *
ellipsis_mask *
shrink_axis_mask*
T0*
Index0
;
ctc/range_1/startConst*
value	B : *
dtype0
;
ctc/range_1/deltaConst*
value	B :*
dtype0
[
ctc/range_1Rangectc/range_1/startctc/strided_slice_5ctc/range_1/delta*

Tidx0
G

ctc/Tile_1Tilectc/range_1ctc/stack_1*

Tmultiples0*
T0
@
ctc/ReverseV2/axisConst*
valueB: *
dtype0
N
ctc/ReverseV2	ReverseV2	ctc/Shapectc/ReverseV2/axis*

Tidx0*
T0
J
ctc/Reshape_1Reshape
ctc/Tile_1ctc/ReverseV2*
Tshape0*
T0
2
ctc/transpose/RankRankctc/Reshape_1*
T0
=
ctc/transpose/sub/yConst*
value	B :*
dtype0
J
ctc/transpose/subSubctc/transpose/Rankctc/transpose/sub/y*
T0
C
ctc/transpose/Range/startConst*
value	B : *
dtype0
C
ctc/transpose/Range/deltaConst*
value	B :*
dtype0
r
ctc/transpose/RangeRangectc/transpose/Range/startctc/transpose/Rankctc/transpose/Range/delta*

Tidx0
K
ctc/transpose/sub_1Subctc/transpose/subctc/transpose/Range*
T0
T
ctc/transpose	Transposectc/Reshape_1ctc/transpose/sub_1*
Tperm0*
T0
I
ctc/boolean_mask_1/ShapeShapectc/transpose*
out_type0*
T0
T
&ctc/boolean_mask_1/strided_slice/stackConst*
valueB: *
dtype0
V
(ctc/boolean_mask_1/strided_slice/stack_1Const*
dtype0*
valueB:
V
(ctc/boolean_mask_1/strided_slice/stack_2Const*
valueB:*
dtype0
Ą
 ctc/boolean_mask_1/strided_sliceStridedSlicectc/boolean_mask_1/Shape&ctc/boolean_mask_1/strided_slice/stack(ctc/boolean_mask_1/strided_slice/stack_1(ctc/boolean_mask_1/strided_slice/stack_2*
shrink_axis_mask *
T0*
Index0*
end_mask *
new_axis_mask *

begin_mask*
ellipsis_mask 
W
)ctc/boolean_mask_1/Prod/reduction_indicesConst*
valueB: *
dtype0

ctc/boolean_mask_1/ProdProd ctc/boolean_mask_1/strided_slice)ctc/boolean_mask_1/Prod/reduction_indices*

Tidx0*
	keep_dims( *
T0
K
ctc/boolean_mask_1/Shape_1Shapectc/transpose*
out_type0*
T0
V
(ctc/boolean_mask_1/strided_slice_1/stackConst*
dtype0*
valueB:
X
*ctc/boolean_mask_1/strided_slice_1/stack_1Const*
valueB: *
dtype0
X
*ctc/boolean_mask_1/strided_slice_1/stack_2Const*
dtype0*
valueB:
Ź
"ctc/boolean_mask_1/strided_slice_1StridedSlicectc/boolean_mask_1/Shape_1(ctc/boolean_mask_1/strided_slice_1/stack*ctc/boolean_mask_1/strided_slice_1/stack_1*ctc/boolean_mask_1/strided_slice_1/stack_2*
end_mask*
new_axis_mask *

begin_mask *
ellipsis_mask *
shrink_axis_mask *
T0*
Index0
a
"ctc/boolean_mask_1/concat/values_0Packctc/boolean_mask_1/Prod*
N*
T0*

axis 
H
ctc/boolean_mask_1/concat/axisConst*
dtype0*
value	B : 
«
ctc/boolean_mask_1/concatConcatV2"ctc/boolean_mask_1/concat/values_0"ctc/boolean_mask_1/strided_slice_1ctc/boolean_mask_1/concat/axis*
N*
T0*

Tidx0
f
ctc/boolean_mask_1/ReshapeReshapectc/transposectc/boolean_mask_1/concat*
Tshape0*
T0
Y
"ctc/boolean_mask_1/Reshape_1/shapeConst*
valueB:
’’’’’’’’’*
dtype0
w
ctc/boolean_mask_1/Reshape_1Reshapectc/strided_slice_3"ctc/boolean_mask_1/Reshape_1/shape*
Tshape0*
T0

?
ctc/boolean_mask_1/WhereWherectc/boolean_mask_1/Reshape_1
_
ctc/boolean_mask_1/SqueezeSqueezectc/boolean_mask_1/Where*
squeeze_dims
*
T0	

ctc/boolean_mask_1/GatherGatherctc/boolean_mask_1/Reshapectc/boolean_mask_1/Squeeze*
Tparams0*
validate_indices(*
Tindices0	
9
ctc/concat/axisConst*
value	B : *
dtype0
y

ctc/concatConcatV2ctc/boolean_mask_1/Gatherctc/boolean_mask/Gatherctc/concat/axis*
N*
T0*

Tidx0
H
ctc/Reshape_2/shapeConst*
valueB"   ’’’’*
dtype0
P
ctc/Reshape_2Reshape
ctc/concatctc/Reshape_2/shape*
T0*
Tshape0
4
ctc/transpose_1/RankRankctc/Reshape_2*
T0
?
ctc/transpose_1/sub/yConst*
value	B :*
dtype0
P
ctc/transpose_1/subSubctc/transpose_1/Rankctc/transpose_1/sub/y*
T0
E
ctc/transpose_1/Range/startConst*
dtype0*
value	B : 
E
ctc/transpose_1/Range/deltaConst*
dtype0*
value	B :
z
ctc/transpose_1/RangeRangectc/transpose_1/Range/startctc/transpose_1/Rankctc/transpose_1/Range/delta*

Tidx0
Q
ctc/transpose_1/sub_1Subctc/transpose_1/subctc/transpose_1/Range*
T0
X
ctc/transpose_1	Transposectc/Reshape_2ctc/transpose_1/sub_1*
Tperm0*
T0
T
ctc/GatherNdGatherNd
the_labelsctc/transpose_1*
Tindices0*
Tparams0
<
ctc/ToInt64Castctc/transpose_1*

DstT0	*

SrcT0
8
ctc/ToInt64_1Cast	ctc/Shape*

DstT0	*

SrcT0
M
ctc/transpose_2/permConst*
dtype0*!
valueB"          
f
ctc/transpose_2	Transposetime_distributed_2/Reshape_1ctc/transpose_2/perm*
Tperm0*
T0
6
	ctc/add/yConst*
dtype0*
valueB
 *wĢ+2
3
ctc/addAddctc/transpose_2	ctc/add/y*
T0
 
ctc/LogLogctc/add*
T0

ctc/CTCLossCTCLossctc/Logctc/ToInt64ctc/GatherNdctc/Squeeze_1*
ctc_merge_repeated(*"
preprocess_collapse_repeated( 
<
ctc/ExpandDims/dimConst*
dtype0*
value	B :
R
ctc/ExpandDims
ExpandDimsctc/CTCLossctc/ExpandDims/dim*
T0*

Tdim0
B
PlaceholderPlaceholder*
dtype0*
shape:”

AssignAssignconv_1/kernelPlaceholder* 
_class
loc:@conv_1/kernel*
T0*
validate_shape(*
use_locking( 
;
Placeholder_1Placeholder*
shape:*
dtype0

Assign_1Assignconv_1/biasPlaceholder_1*
validate_shape(*
_class
loc:@conv_1/bias*
T0*
use_locking( 
@
Placeholder_2Placeholder*
shape:
*
dtype0

Assign_2Assigntime_distributed_1/kernelPlaceholder_2*
use_locking( *
validate_shape(*
T0*,
_class"
 loc:@time_distributed_1/kernel
;
Placeholder_3Placeholder*
dtype0*
shape:

Assign_3Assigntime_distributed_1/biasPlaceholder_3*
use_locking( *
T0**
_class 
loc:@time_distributed_1/bias*
validate_shape(
?
Placeholder_4Placeholder*
dtype0*
shape:	

Assign_4Assigntime_distributed_2/kernelPlaceholder_4*
validate_shape(*,
_class"
 loc:@time_distributed_2/kernel*
T0*
use_locking( 
:
Placeholder_5Placeholder*
shape:*
dtype0

Assign_5Assigntime_distributed_2/biasPlaceholder_5*
use_locking( *
T0**
_class 
loc:@time_distributed_2/bias*
validate_shape(
Ą
initNoOp^conv_1/kernel/Assign^conv_1/bias/Assign!^time_distributed_1/kernel/Assign^time_distributed_1/bias/Assign!^time_distributed_2/kernel/Assign^time_distributed_2/bias/Assign
E
iterations/initial_valueConst*
valueB
 *    *
dtype0
V

iterations
VariableV2*
shared_name *
dtype0*
shape: *
	container 

iterations/AssignAssign
iterationsiterations/initial_value*
use_locking(*
validate_shape(*
T0*
_class
loc:@iterations
O
iterations/readIdentity
iterations*
_class
loc:@iterations*
T0
=
lr/initial_valueConst*
valueB
 *
×#<*
dtype0
N
lr
VariableV2*
shared_name *
dtype0*
shape: *
	container 
r
	lr/AssignAssignlrlr/initial_value*
use_locking(*
validate_shape(*
T0*
_class
	loc:@lr
7
lr/readIdentitylr*
_class
	loc:@lr*
T0
C
momentum/initial_valueConst*
valueB
 *fff?*
dtype0
T
momentum
VariableV2*
shared_name *
dtype0*
shape: *
	container 

momentum/AssignAssignmomentummomentum/initial_value*
validate_shape(*
_class
loc:@momentum*
T0*
use_locking(
I
momentum/readIdentitymomentum*
_class
loc:@momentum*
T0
@
decay/initial_valueConst*
valueB
 *½75*
dtype0
Q
decay
VariableV2*
	container *
dtype0*
shared_name *
shape: 
~
decay/AssignAssigndecaydecay/initial_value*
use_locking(*
T0*
_class

loc:@decay*
validate_shape(
@

decay/readIdentitydecay*
_class

loc:@decay*
T0
;
ctc_sample_weightsPlaceholder*
shape: *
dtype0
3

ctc_targetPlaceholder*
dtype0*
shape: 
D
Mean/reduction_indicesConst*
valueB:*
dtype0
Z
MeanMeanctc/ExpandDimsMean/reduction_indices*

Tidx0*
	keep_dims( *
T0
-
mulMulMeanctc_sample_weights*
T0
7

NotEqual/yConst*
valueB
 *    *
dtype0
=
NotEqualNotEqualctc_sample_weights
NotEqual/y*
T0
.
CastCastNotEqual*

DstT0*

SrcT0

3
ConstConst*
dtype0*
valueB: 
A
Mean_1MeanCastConst*

Tidx0*
	keep_dims( *
T0
$
divRealDivmulMean_1*
T0
5
Const_1Const*
valueB: *
dtype0
B
Mean_2MeandivConst_1*
T0*

Tidx0*
	keep_dims( 
4
mul_1/xConst*
valueB
 *  ?*
dtype0
&
mul_1Mulmul_1/xMean_2*
T0
4
the_input_1Placeholder*
shape: *
dtype0
h
zero_padding1d_1_1/Pad/paddingsConst*
dtype0*1
value(B&"                       
e
zero_padding1d_1_1/PadPadthe_input_1zero_padding1d_1_1/Pad/paddings*
T0*
	Tpaddings0
O
init_1NoOp^iterations/Assign
^lr/Assign^momentum/Assign^decay/Assign
V
conv_1_1/random_uniform/shapeConst*!
valueB"   ”      *
dtype0
H
conv_1_1/random_uniform/minConst*
dtype0*
valueB
 *
»
H
conv_1_1/random_uniform/maxConst*
valueB
 *
;*
dtype0

%conv_1_1/random_uniform/RandomUniformRandomUniformconv_1_1/random_uniform/shape*
seed±’å)*
T0*
dtype0*
seed2ø’r
e
conv_1_1/random_uniform/subSubconv_1_1/random_uniform/maxconv_1_1/random_uniform/min*
T0
o
conv_1_1/random_uniform/mulMul%conv_1_1/random_uniform/RandomUniformconv_1_1/random_uniform/sub*
T0
a
conv_1_1/random_uniformAddconv_1_1/random_uniform/mulconv_1_1/random_uniform/min*
T0
i
conv_1_1/kernel
VariableV2*
shared_name *
dtype0*
shape:”*
	container 
 
conv_1_1/kernel/AssignAssignconv_1_1/kernelconv_1_1/random_uniform*
validate_shape(*"
_class
loc:@conv_1_1/kernel*
T0*
use_locking(
^
conv_1_1/kernel/readIdentityconv_1_1/kernel*"
_class
loc:@conv_1_1/kernel*
T0
@
conv_1_1/ConstConst*
dtype0*
valueB*    
^
conv_1_1/bias
VariableV2*
	container *
shape:*
dtype0*
shared_name 

conv_1_1/bias/AssignAssignconv_1_1/biasconv_1_1/Const*
use_locking(*
T0* 
_class
loc:@conv_1_1/bias*
validate_shape(
X
conv_1_1/bias/readIdentityconv_1_1/bias*
T0* 
_class
loc:@conv_1_1/bias
E
conv_1_1/initNoOp^conv_1_1/kernel/Assign^conv_1_1/bias/Assign
K
conv_1_1/PlaceholderPlaceholder*
dtype0*
shape:”

conv_1_1/AssignAssignconv_1_1/kernelconv_1_1/Placeholder*
use_locking( *
T0*"
_class
loc:@conv_1_1/kernel*
validate_shape(
D
conv_1_1/Placeholder_1Placeholder*
shape:*
dtype0

conv_1_1/Assign_1Assignconv_1_1/biasconv_1_1/Placeholder_1* 
_class
loc:@conv_1_1/bias*
T0*
validate_shape(*
use_locking( 
S
conv_1_1/convolution/ShapeConst*!
valueB"   ”      *
dtype0
P
"conv_1_1/convolution/dilation_rateConst*
dtype0*
valueB:
M
#conv_1_1/convolution/ExpandDims/dimConst*
dtype0*
value	B :

conv_1_1/convolution/ExpandDims
ExpandDimszero_padding1d_1_1/Pad#conv_1_1/convolution/ExpandDims/dim*

Tdim0*
T0
O
%conv_1_1/convolution/ExpandDims_1/dimConst*
value	B : *
dtype0

!conv_1_1/convolution/ExpandDims_1
ExpandDimsconv_1_1/kernel/read%conv_1_1/convolution/ExpandDims_1/dim*
T0*

Tdim0
Ź
conv_1_1/convolution/Conv2DConv2Dconv_1_1/convolution/ExpandDims!conv_1_1/convolution/ExpandDims_1*
paddingVALID*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
d
conv_1_1/convolution/SqueezeSqueezeconv_1_1/convolution/Conv2D*
squeeze_dims
*
T0
O
conv_1_1/Reshape/shapeConst*
dtype0*!
valueB"         
^
conv_1_1/ReshapeReshapeconv_1_1/bias/readconv_1_1/Reshape/shape*
Tshape0*
T0
L
conv_1_1/addAddconv_1_1/convolution/Squeezeconv_1_1/Reshape*
T0
,
conv_1_1/ReluReluconv_1_1/add*
T0
^
)time_distributed_1_1/random_uniform/shapeConst*
valueB"      *
dtype0
T
'time_distributed_1_1/random_uniform/minConst*
dtype0*
valueB
 *qÄ½
T
'time_distributed_1_1/random_uniform/maxConst*
dtype0*
valueB
 *qÄ=
¢
1time_distributed_1_1/random_uniform/RandomUniformRandomUniform)time_distributed_1_1/random_uniform/shape*
seed2Ļ¼*
dtype0*
T0*
seed±’å)

'time_distributed_1_1/random_uniform/subSub'time_distributed_1_1/random_uniform/max'time_distributed_1_1/random_uniform/min*
T0

'time_distributed_1_1/random_uniform/mulMul1time_distributed_1_1/random_uniform/RandomUniform'time_distributed_1_1/random_uniform/sub*
T0

#time_distributed_1_1/random_uniformAdd'time_distributed_1_1/random_uniform/mul'time_distributed_1_1/random_uniform/min*
T0
q
time_distributed_1_1/kernel
VariableV2*
shape:
*
shared_name *
dtype0*
	container 
Š
"time_distributed_1_1/kernel/AssignAssigntime_distributed_1_1/kernel#time_distributed_1_1/random_uniform*.
_class$
" loc:@time_distributed_1_1/kernel*
T0*
validate_shape(*
use_locking(

 time_distributed_1_1/kernel/readIdentitytime_distributed_1_1/kernel*
T0*.
_class$
" loc:@time_distributed_1_1/kernel
L
time_distributed_1_1/ConstConst*
valueB*    *
dtype0
j
time_distributed_1_1/bias
VariableV2*
shared_name *
dtype0*
shape:*
	container 
Į
 time_distributed_1_1/bias/AssignAssigntime_distributed_1_1/biastime_distributed_1_1/Const*,
_class"
 loc:@time_distributed_1_1/bias*
T0*
validate_shape(*
use_locking(
|
time_distributed_1_1/bias/readIdentitytime_distributed_1_1/bias*
T0*,
_class"
 loc:@time_distributed_1_1/bias
K
time_distributed_1_1/ShapeShapeconv_1_1/Relu*
out_type0*
T0
V
(time_distributed_1_1/strided_slice/stackConst*
dtype0*
valueB:
X
*time_distributed_1_1/strided_slice/stack_1Const*
valueB:*
dtype0
X
*time_distributed_1_1/strided_slice/stack_2Const*
valueB:*
dtype0
Ź
"time_distributed_1_1/strided_sliceStridedSlicetime_distributed_1_1/Shape(time_distributed_1_1/strided_slice/stack*time_distributed_1_1/strided_slice/stack_1*time_distributed_1_1/strided_slice/stack_2*
new_axis_mask *
shrink_axis_mask*
T0*
Index0*
end_mask *

begin_mask *
ellipsis_mask 
W
"time_distributed_1_1/Reshape/shapeConst*
valueB"’’’’   *
dtype0
q
time_distributed_1_1/ReshapeReshapeconv_1_1/Relu"time_distributed_1_1/Reshape/shape*
Tshape0*
T0

time_distributed_1_1/MatMulMatMultime_distributed_1_1/Reshape time_distributed_1_1/kernel/read*
transpose_b( *
transpose_a( *
T0

time_distributed_1_1/BiasAddBiasAddtime_distributed_1_1/MatMultime_distributed_1_1/bias/read*
T0*
data_formatNHWC
H
time_distributed_1_1/ReluRelutime_distributed_1_1/BiasAdd*
T0
Y
&time_distributed_1_1/Reshape_1/shape/0Const*
dtype0*
valueB :
’’’’’’’’’
Q
&time_distributed_1_1/Reshape_1/shape/2Const*
dtype0*
value
B :
¾
$time_distributed_1_1/Reshape_1/shapePack&time_distributed_1_1/Reshape_1/shape/0"time_distributed_1_1/strided_slice&time_distributed_1_1/Reshape_1/shape/2*
T0*

axis *
N

time_distributed_1_1/Reshape_1Reshapetime_distributed_1_1/Relu$time_distributed_1_1/Reshape_1/shape*
Tshape0*
T0
^
)time_distributed_2_1/random_uniform/shapeConst*
valueB"      *
dtype0
T
'time_distributed_2_1/random_uniform/minConst*
dtype0*
valueB
 *µ­×½
T
'time_distributed_2_1/random_uniform/maxConst*
dtype0*
valueB
 *µ­×=
¢
1time_distributed_2_1/random_uniform/RandomUniformRandomUniform)time_distributed_2_1/random_uniform/shape*
seed±’å)*
T0*
dtype0*
seed2č

'time_distributed_2_1/random_uniform/subSub'time_distributed_2_1/random_uniform/max'time_distributed_2_1/random_uniform/min*
T0

'time_distributed_2_1/random_uniform/mulMul1time_distributed_2_1/random_uniform/RandomUniform'time_distributed_2_1/random_uniform/sub*
T0

#time_distributed_2_1/random_uniformAdd'time_distributed_2_1/random_uniform/mul'time_distributed_2_1/random_uniform/min*
T0
p
time_distributed_2_1/kernel
VariableV2*
	container *
shape:	*
dtype0*
shared_name 
Š
"time_distributed_2_1/kernel/AssignAssigntime_distributed_2_1/kernel#time_distributed_2_1/random_uniform*
use_locking(*
validate_shape(*
T0*.
_class$
" loc:@time_distributed_2_1/kernel

 time_distributed_2_1/kernel/readIdentitytime_distributed_2_1/kernel*
T0*.
_class$
" loc:@time_distributed_2_1/kernel
K
time_distributed_2_1/ConstConst*
dtype0*
valueB*    
i
time_distributed_2_1/bias
VariableV2*
	container *
shape:*
dtype0*
shared_name 
Į
 time_distributed_2_1/bias/AssignAssigntime_distributed_2_1/biastime_distributed_2_1/Const*,
_class"
 loc:@time_distributed_2_1/bias*
T0*
validate_shape(*
use_locking(
|
time_distributed_2_1/bias/readIdentitytime_distributed_2_1/bias*,
_class"
 loc:@time_distributed_2_1/bias*
T0
\
time_distributed_2_1/ShapeShapetime_distributed_1_1/Reshape_1*
out_type0*
T0
V
(time_distributed_2_1/strided_slice/stackConst*
valueB:*
dtype0
X
*time_distributed_2_1/strided_slice/stack_1Const*
valueB:*
dtype0
X
*time_distributed_2_1/strided_slice/stack_2Const*
valueB:*
dtype0
Ź
"time_distributed_2_1/strided_sliceStridedSlicetime_distributed_2_1/Shape(time_distributed_2_1/strided_slice/stack*time_distributed_2_1/strided_slice/stack_1*time_distributed_2_1/strided_slice/stack_2*
end_mask *

begin_mask *
ellipsis_mask *
shrink_axis_mask*
new_axis_mask *
T0*
Index0
W
"time_distributed_2_1/Reshape/shapeConst*
dtype0*
valueB"’’’’   

time_distributed_2_1/ReshapeReshapetime_distributed_1_1/Reshape_1"time_distributed_2_1/Reshape/shape*
T0*
Tshape0

time_distributed_2_1/MatMulMatMultime_distributed_2_1/Reshape time_distributed_2_1/kernel/read*
transpose_b( *
T0*
transpose_a( 

time_distributed_2_1/BiasAddBiasAddtime_distributed_2_1/MatMultime_distributed_2_1/bias/read*
data_formatNHWC*
T0
N
time_distributed_2_1/SoftmaxSoftmaxtime_distributed_2_1/BiasAdd*
T0
Y
&time_distributed_2_1/Reshape_1/shape/0Const*
dtype0*
valueB :
’’’’’’’’’
P
&time_distributed_2_1/Reshape_1/shape/2Const*
value	B :*
dtype0
¾
$time_distributed_2_1/Reshape_1/shapePack&time_distributed_2_1/Reshape_1/shape/0"time_distributed_2_1/strided_slice&time_distributed_2_1/Reshape_1/shape/2*
N*
T0*

axis 

time_distributed_2_1/Reshape_1Reshapetime_distributed_2_1/Softmax$time_distributed_2_1/Reshape_1/shape*
T0*
Tshape0
G
iterations_1/initial_valueConst*
valueB
 *    *
dtype0
X
iterations_1
VariableV2*
shape: *
shared_name *
dtype0*
	container 

iterations_1/AssignAssigniterations_1iterations_1/initial_value*
use_locking(*
T0*
_class
loc:@iterations_1*
validate_shape(
U
iterations_1/readIdentityiterations_1*
T0*
_class
loc:@iterations_1
?
lr_1/initial_valueConst*
dtype0*
valueB
 *
×£<
P
lr_1
VariableV2*
	container *
shape: *
dtype0*
shared_name 
z
lr_1/AssignAssignlr_1lr_1/initial_value*
_class
	loc:@lr_1*
T0*
validate_shape(*
use_locking(
=
	lr_1/readIdentitylr_1*
T0*
_class
	loc:@lr_1
E
momentum_1/initial_valueConst*
dtype0*
valueB
 *fff?
V

momentum_1
VariableV2*
shared_name *
dtype0*
shape: *
	container 

momentum_1/AssignAssign
momentum_1momentum_1/initial_value*
_class
loc:@momentum_1*
T0*
validate_shape(*
use_locking(
O
momentum_1/readIdentity
momentum_1*
_class
loc:@momentum_1*
T0
B
decay_1/initial_valueConst*
dtype0*
valueB
 *½75
S
decay_1
VariableV2*
shape: *
shared_name *
dtype0*
	container 

decay_1/AssignAssigndecay_1decay_1/initial_value*
_class
loc:@decay_1*
T0*
validate_shape(*
use_locking(
F
decay_1/readIdentitydecay_1*
_class
loc:@decay_1*
T0
J
!time_distributed_2_sample_weightsPlaceholder*
shape: *
dtype0
B
time_distributed_2_targetPlaceholder*
shape: *
dtype0
N
subSubtime_distributed_2_1/Reshape_1time_distributed_2_target*
T0

SquareSquaresub*
T0
B
Mean_3/reduction_indicesConst*
value	B :*
dtype0
V
Mean_3MeanSquareMean_3/reduction_indices*
T0*

Tidx0*
	keep_dims( 
F
Mean_4/reduction_indicesConst*
valueB:*
dtype0
V
Mean_4MeanMean_3Mean_4/reduction_indices*
T0*

Tidx0*
	keep_dims( 
@
mul_2MulMean_4!time_distributed_2_sample_weights*
T0
9
NotEqual_1/yConst*
valueB
 *    *
dtype0
P

NotEqual_1NotEqual!time_distributed_2_sample_weightsNotEqual_1/y*
T0
2
Cast_1Cast
NotEqual_1*

SrcT0
*

DstT0
5
Const_2Const*
valueB: *
dtype0
E
Mean_5MeanCast_1Const_2*
T0*

Tidx0*
	keep_dims( 
(
div_1RealDivmul_2Mean_5*
T0
5
Const_3Const*
valueB: *
dtype0
D
Mean_6Meandiv_1Const_3*

Tidx0*
	keep_dims( *
T0
4
mul_3/xConst*
dtype0*
valueB
 *  ?
&
mul_3Mulmul_3/xMean_6*
T0
ē
init_2NoOp#^time_distributed_1_1/kernel/Assign!^time_distributed_1_1/bias/Assign#^time_distributed_2_1/kernel/Assign!^time_distributed_2_1/bias/Assign^iterations_1/Assign^lr_1/Assign^momentum_1/Assign^decay_1/Assign
@
Placeholder_6Placeholder*
shape:
*
dtype0
 
Assign_6Assigntime_distributed_1_1/kernelPlaceholder_6*.
_class$
" loc:@time_distributed_1_1/kernel*
T0*
validate_shape(*
use_locking( 
;
Placeholder_7Placeholder*
dtype0*
shape:

Assign_7Assigntime_distributed_1_1/biasPlaceholder_7*
validate_shape(*,
_class"
 loc:@time_distributed_1_1/bias*
T0*
use_locking( 
?
Placeholder_8Placeholder*
shape:	*
dtype0
 
Assign_8Assigntime_distributed_2_1/kernelPlaceholder_8*
use_locking( *
T0*.
_class$
" loc:@time_distributed_2_1/kernel*
validate_shape(
:
Placeholder_9Placeholder*
shape:*
dtype0

Assign_9Assigntime_distributed_2_1/biasPlaceholder_9*
validate_shape(*,
_class"
 loc:@time_distributed_2_1/bias*
T0*
use_locking( 

init_3NoOp^conv_1/kernel/Assign^conv_1/bias/Assign!^time_distributed_1/kernel/Assign^time_distributed_1/bias/Assign!^time_distributed_2/kernel/Assign^time_distributed_2/bias/Assign^iterations/Assign
^lr/Assign^momentum/Assign^decay/Assign^conv_1_1/kernel/Assign^conv_1_1/bias/Assign#^time_distributed_1_1/kernel/Assign!^time_distributed_1_1/bias/Assign#^time_distributed_2_1/kernel/Assign!^time_distributed_2_1/bias/Assign^iterations_1/Assign^lr_1/Assign^momentum_1/Assign^decay_1/Assign
8

save/ConstConst*
valueB Bmodel*
dtype0
¤
save/SaveV2/tensor_namesConst*
dtype0*ó
valueéBęBconv_1/biasBconv_1/kernelBconv_1_1/biasBconv_1_1/kernelBdecayBdecay_1B
iterationsBiterations_1BlrBlr_1BmomentumB
momentum_1Btime_distributed_1/biasBtime_distributed_1/kernelBtime_distributed_1_1/biasBtime_distributed_1_1/kernelBtime_distributed_2/biasBtime_distributed_2/kernelBtime_distributed_2_1/biasBtime_distributed_2_1/kernel
o
save/SaveV2/shape_and_slicesConst*;
value2B0B B B B B B B B B B B B B B B B B B B B *
dtype0
Ū
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesconv_1/biasconv_1/kernelconv_1_1/biasconv_1_1/kerneldecaydecay_1
iterationsiterations_1lrlr_1momentum
momentum_1time_distributed_1/biastime_distributed_1/kerneltime_distributed_1_1/biastime_distributed_1_1/kerneltime_distributed_2/biastime_distributed_2/kerneltime_distributed_2_1/biastime_distributed_2_1/kernel*"
dtypes
2
e
save/control_dependencyIdentity
save/Const^save/SaveV2*
T0*
_class
loc:@save/Const
S
save/RestoreV2/tensor_namesConst*
dtype0* 
valueBBconv_1/bias
L
save/RestoreV2/shape_and_slicesConst*
dtype0*
valueB
B 
v
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*
dtypes
2

save/AssignAssignconv_1/biassave/RestoreV2*
use_locking(*
T0*
_class
loc:@conv_1/bias*
validate_shape(
W
save/RestoreV2_1/tensor_namesConst*"
valueBBconv_1/kernel*
dtype0
N
!save/RestoreV2_1/shape_and_slicesConst*
valueB
B *
dtype0
|
save/RestoreV2_1	RestoreV2
save/Constsave/RestoreV2_1/tensor_names!save/RestoreV2_1/shape_and_slices*
dtypes
2

save/Assign_1Assignconv_1/kernelsave/RestoreV2_1*
use_locking(*
T0* 
_class
loc:@conv_1/kernel*
validate_shape(
W
save/RestoreV2_2/tensor_namesConst*
dtype0*"
valueBBconv_1_1/bias
N
!save/RestoreV2_2/shape_and_slicesConst*
valueB
B *
dtype0
|
save/RestoreV2_2	RestoreV2
save/Constsave/RestoreV2_2/tensor_names!save/RestoreV2_2/shape_and_slices*
dtypes
2

save/Assign_2Assignconv_1_1/biassave/RestoreV2_2*
use_locking(*
T0* 
_class
loc:@conv_1_1/bias*
validate_shape(
Y
save/RestoreV2_3/tensor_namesConst*
dtype0*$
valueBBconv_1_1/kernel
N
!save/RestoreV2_3/shape_and_slicesConst*
valueB
B *
dtype0
|
save/RestoreV2_3	RestoreV2
save/Constsave/RestoreV2_3/tensor_names!save/RestoreV2_3/shape_and_slices*
dtypes
2

save/Assign_3Assignconv_1_1/kernelsave/RestoreV2_3*
use_locking(*
validate_shape(*
T0*"
_class
loc:@conv_1_1/kernel
O
save/RestoreV2_4/tensor_namesConst*
valueBBdecay*
dtype0
N
!save/RestoreV2_4/shape_and_slicesConst*
valueB
B *
dtype0
|
save/RestoreV2_4	RestoreV2
save/Constsave/RestoreV2_4/tensor_names!save/RestoreV2_4/shape_and_slices*
dtypes
2
|
save/Assign_4Assigndecaysave/RestoreV2_4*
use_locking(*
T0*
_class

loc:@decay*
validate_shape(
Q
save/RestoreV2_5/tensor_namesConst*
dtype0*
valueBBdecay_1
N
!save/RestoreV2_5/shape_and_slicesConst*
dtype0*
valueB
B 
|
save/RestoreV2_5	RestoreV2
save/Constsave/RestoreV2_5/tensor_names!save/RestoreV2_5/shape_and_slices*
dtypes
2

save/Assign_5Assigndecay_1save/RestoreV2_5*
use_locking(*
validate_shape(*
T0*
_class
loc:@decay_1
T
save/RestoreV2_6/tensor_namesConst*
dtype0*
valueBB
iterations
N
!save/RestoreV2_6/shape_and_slicesConst*
valueB
B *
dtype0
|
save/RestoreV2_6	RestoreV2
save/Constsave/RestoreV2_6/tensor_names!save/RestoreV2_6/shape_and_slices*
dtypes
2

save/Assign_6Assign
iterationssave/RestoreV2_6*
use_locking(*
validate_shape(*
T0*
_class
loc:@iterations
V
save/RestoreV2_7/tensor_namesConst*
dtype0*!
valueBBiterations_1
N
!save/RestoreV2_7/shape_and_slicesConst*
dtype0*
valueB
B 
|
save/RestoreV2_7	RestoreV2
save/Constsave/RestoreV2_7/tensor_names!save/RestoreV2_7/shape_and_slices*
dtypes
2

save/Assign_7Assigniterations_1save/RestoreV2_7*
_class
loc:@iterations_1*
T0*
validate_shape(*
use_locking(
L
save/RestoreV2_8/tensor_namesConst*
valueBBlr*
dtype0
N
!save/RestoreV2_8/shape_and_slicesConst*
valueB
B *
dtype0
|
save/RestoreV2_8	RestoreV2
save/Constsave/RestoreV2_8/tensor_names!save/RestoreV2_8/shape_and_slices*
dtypes
2
v
save/Assign_8Assignlrsave/RestoreV2_8*
use_locking(*
validate_shape(*
T0*
_class
	loc:@lr
N
save/RestoreV2_9/tensor_namesConst*
valueBBlr_1*
dtype0
N
!save/RestoreV2_9/shape_and_slicesConst*
dtype0*
valueB
B 
|
save/RestoreV2_9	RestoreV2
save/Constsave/RestoreV2_9/tensor_names!save/RestoreV2_9/shape_and_slices*
dtypes
2
z
save/Assign_9Assignlr_1save/RestoreV2_9*
_class
	loc:@lr_1*
T0*
validate_shape(*
use_locking(
S
save/RestoreV2_10/tensor_namesConst*
dtype0*
valueBBmomentum
O
"save/RestoreV2_10/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_10	RestoreV2
save/Constsave/RestoreV2_10/tensor_names"save/RestoreV2_10/shape_and_slices*
dtypes
2

save/Assign_10Assignmomentumsave/RestoreV2_10*
validate_shape(*
_class
loc:@momentum*
T0*
use_locking(
U
save/RestoreV2_11/tensor_namesConst*
dtype0*
valueBB
momentum_1
O
"save/RestoreV2_11/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_11	RestoreV2
save/Constsave/RestoreV2_11/tensor_names"save/RestoreV2_11/shape_and_slices*
dtypes
2

save/Assign_11Assign
momentum_1save/RestoreV2_11*
validate_shape(*
_class
loc:@momentum_1*
T0*
use_locking(
b
save/RestoreV2_12/tensor_namesConst*,
value#B!Btime_distributed_1/bias*
dtype0
O
"save/RestoreV2_12/shape_and_slicesConst*
dtype0*
valueB
B 

save/RestoreV2_12	RestoreV2
save/Constsave/RestoreV2_12/tensor_names"save/RestoreV2_12/shape_and_slices*
dtypes
2
¢
save/Assign_12Assigntime_distributed_1/biassave/RestoreV2_12*
use_locking(*
validate_shape(*
T0**
_class 
loc:@time_distributed_1/bias
d
save/RestoreV2_13/tensor_namesConst*.
value%B#Btime_distributed_1/kernel*
dtype0
O
"save/RestoreV2_13/shape_and_slicesConst*
dtype0*
valueB
B 

save/RestoreV2_13	RestoreV2
save/Constsave/RestoreV2_13/tensor_names"save/RestoreV2_13/shape_and_slices*
dtypes
2
¦
save/Assign_13Assigntime_distributed_1/kernelsave/RestoreV2_13*
use_locking(*
T0*,
_class"
 loc:@time_distributed_1/kernel*
validate_shape(
d
save/RestoreV2_14/tensor_namesConst*
dtype0*.
value%B#Btime_distributed_1_1/bias
O
"save/RestoreV2_14/shape_and_slicesConst*
dtype0*
valueB
B 

save/RestoreV2_14	RestoreV2
save/Constsave/RestoreV2_14/tensor_names"save/RestoreV2_14/shape_and_slices*
dtypes
2
¦
save/Assign_14Assigntime_distributed_1_1/biassave/RestoreV2_14*
use_locking(*
validate_shape(*
T0*,
_class"
 loc:@time_distributed_1_1/bias
f
save/RestoreV2_15/tensor_namesConst*
dtype0*0
value'B%Btime_distributed_1_1/kernel
O
"save/RestoreV2_15/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_15	RestoreV2
save/Constsave/RestoreV2_15/tensor_names"save/RestoreV2_15/shape_and_slices*
dtypes
2
Ŗ
save/Assign_15Assigntime_distributed_1_1/kernelsave/RestoreV2_15*
use_locking(*
T0*.
_class$
" loc:@time_distributed_1_1/kernel*
validate_shape(
b
save/RestoreV2_16/tensor_namesConst*
dtype0*,
value#B!Btime_distributed_2/bias
O
"save/RestoreV2_16/shape_and_slicesConst*
dtype0*
valueB
B 

save/RestoreV2_16	RestoreV2
save/Constsave/RestoreV2_16/tensor_names"save/RestoreV2_16/shape_and_slices*
dtypes
2
¢
save/Assign_16Assigntime_distributed_2/biassave/RestoreV2_16*
use_locking(*
validate_shape(*
T0**
_class 
loc:@time_distributed_2/bias
d
save/RestoreV2_17/tensor_namesConst*
dtype0*.
value%B#Btime_distributed_2/kernel
O
"save/RestoreV2_17/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_17	RestoreV2
save/Constsave/RestoreV2_17/tensor_names"save/RestoreV2_17/shape_and_slices*
dtypes
2
¦
save/Assign_17Assigntime_distributed_2/kernelsave/RestoreV2_17*
validate_shape(*,
_class"
 loc:@time_distributed_2/kernel*
T0*
use_locking(
d
save/RestoreV2_18/tensor_namesConst*.
value%B#Btime_distributed_2_1/bias*
dtype0
O
"save/RestoreV2_18/shape_and_slicesConst*
dtype0*
valueB
B 

save/RestoreV2_18	RestoreV2
save/Constsave/RestoreV2_18/tensor_names"save/RestoreV2_18/shape_and_slices*
dtypes
2
¦
save/Assign_18Assigntime_distributed_2_1/biassave/RestoreV2_18*
validate_shape(*,
_class"
 loc:@time_distributed_2_1/bias*
T0*
use_locking(
f
save/RestoreV2_19/tensor_namesConst*
dtype0*0
value'B%Btime_distributed_2_1/kernel
O
"save/RestoreV2_19/shape_and_slicesConst*
valueB
B *
dtype0

save/RestoreV2_19	RestoreV2
save/Constsave/RestoreV2_19/tensor_names"save/RestoreV2_19/shape_and_slices*
dtypes
2
Ŗ
save/Assign_19Assigntime_distributed_2_1/kernelsave/RestoreV2_19*
validate_shape(*.
_class$
" loc:@time_distributed_2_1/kernel*
T0*
use_locking(
ą
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19"