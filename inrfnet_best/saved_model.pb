??<
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
?
AvgPool

value"T
output"T"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW"
Ttype:
2
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

?
ExtractImagePatches
images"T
patches"T"
ksizes	list(int)(0"
strides	list(int)(0"
rates	list(int)(0"
Ttype:
2	
""
paddingstring:
SAMEVALID
?
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%??8"&
exponential_avg_factorfloat%  ??";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
?
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.6.02v2.6.0-rc2-32-g919f693420e8??5
u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?$@*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	?$@*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:@*
dtype0
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

:@*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
~
conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/kernel
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:*
dtype0
n
conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/bias
g
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes
:*
dtype0
?
batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namebatch_normalization/gamma
?
-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes
:*
dtype0
?
batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_namebatch_normalization/beta
?
,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes
:*
dtype0
?
conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	* 
shared_nameconv2d_1/kernel
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
:	*
dtype0
r
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_nameconv2d_1/bias
k
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes
:	*
dtype0
?
conv2d/kernel_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d/kernel_1
{
#conv2d/kernel_1/Read/ReadVariableOpReadVariableOpconv2d/kernel_1*&
_output_shapes
:*
dtype0
r
conv2d/bias_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/bias_1
k
!conv2d/bias_1/Read/ReadVariableOpReadVariableOpconv2d/bias_1*
_output_shapes
:*
dtype0
?
batch_normalization/gamma_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization/gamma_1
?
/batch_normalization/gamma_1/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma_1*
_output_shapes
:*
dtype0
?
batch_normalization/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization/beta_1
?
.batch_normalization/beta_1/Read/ReadVariableOpReadVariableOpbatch_normalization/beta_1*
_output_shapes
:*
dtype0
?
conv2d_1/kernel_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameconv2d_1/kernel_1

%conv2d_1/kernel_1/Read/ReadVariableOpReadVariableOpconv2d_1/kernel_1*&
_output_shapes
:*
dtype0
v
conv2d_1/bias_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_1/bias_1
o
#conv2d_1/bias_1/Read/ReadVariableOpReadVariableOpconv2d_1/bias_1*
_output_shapes
:*
dtype0
?
conv2d/kernel_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d/kernel_2
{
#conv2d/kernel_2/Read/ReadVariableOpReadVariableOpconv2d/kernel_2*&
_output_shapes
:*
dtype0
r
conv2d/bias_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/bias_2
k
!conv2d/bias_2/Read/ReadVariableOpReadVariableOpconv2d/bias_2*
_output_shapes
:*
dtype0
?
batch_normalization/gamma_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization/gamma_2
?
/batch_normalization/gamma_2/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma_2*
_output_shapes
:*
dtype0
?
batch_normalization/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization/beta_2
?
.batch_normalization/beta_2/Read/ReadVariableOpReadVariableOpbatch_normalization/beta_2*
_output_shapes
:*
dtype0
?
conv2d_1/kernel_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameconv2d_1/kernel_2

%conv2d_1/kernel_2/Read/ReadVariableOpReadVariableOpconv2d_1/kernel_2*&
_output_shapes
:*
dtype0
v
conv2d_1/bias_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_1/bias_2
o
#conv2d_1/bias_2/Read/ReadVariableOpReadVariableOpconv2d_1/bias_2*
_output_shapes
:*
dtype0
?
conv2d/kernel_3VarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d/kernel_3
{
#conv2d/kernel_3/Read/ReadVariableOpReadVariableOpconv2d/kernel_3*&
_output_shapes
:*
dtype0
r
conv2d/bias_3VarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/bias_3
k
!conv2d/bias_3/Read/ReadVariableOpReadVariableOpconv2d/bias_3*
_output_shapes
:*
dtype0
?
batch_normalization/gamma_3VarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization/gamma_3
?
/batch_normalization/gamma_3/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma_3*
_output_shapes
:*
dtype0
?
batch_normalization/beta_3VarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization/beta_3
?
.batch_normalization/beta_3/Read/ReadVariableOpReadVariableOpbatch_normalization/beta_3*
_output_shapes
:*
dtype0
?
conv2d_1/kernel_3VarHandleOp*
_output_shapes
: *
dtype0*
shape:	*"
shared_nameconv2d_1/kernel_3

%conv2d_1/kernel_3/Read/ReadVariableOpReadVariableOpconv2d_1/kernel_3*&
_output_shapes
:	*
dtype0
v
conv2d_1/bias_3VarHandleOp*
_output_shapes
: *
dtype0*
shape:	* 
shared_nameconv2d_1/bias_3
o
#conv2d_1/bias_3/Read/ReadVariableOpReadVariableOpconv2d_1/bias_3*
_output_shapes
:	*
dtype0
?
batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!batch_normalization/moving_mean
?
3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes
:*
dtype0
?
#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#batch_normalization/moving_variance
?
7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes
:*
dtype0
?
!batch_normalization/moving_mean_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization/moving_mean_1
?
5batch_normalization/moving_mean_1/Read/ReadVariableOpReadVariableOp!batch_normalization/moving_mean_1*
_output_shapes
:*
dtype0
?
%batch_normalization/moving_variance_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization/moving_variance_1
?
9batch_normalization/moving_variance_1/Read/ReadVariableOpReadVariableOp%batch_normalization/moving_variance_1*
_output_shapes
:*
dtype0
?
!batch_normalization/moving_mean_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization/moving_mean_2
?
5batch_normalization/moving_mean_2/Read/ReadVariableOpReadVariableOp!batch_normalization/moving_mean_2*
_output_shapes
:*
dtype0
?
%batch_normalization/moving_variance_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization/moving_variance_2
?
9batch_normalization/moving_variance_2/Read/ReadVariableOpReadVariableOp%batch_normalization/moving_variance_2*
_output_shapes
:*
dtype0
?
!batch_normalization/moving_mean_3VarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization/moving_mean_3
?
5batch_normalization/moving_mean_3/Read/ReadVariableOpReadVariableOp!batch_normalization/moving_mean_3*
_output_shapes
:*
dtype0
?
%batch_normalization/moving_variance_3VarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization/moving_variance_3
?
9batch_normalization/moving_variance_3/Read/ReadVariableOpReadVariableOp%batch_normalization/moving_variance_3*
_output_shapes
:*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?$@*$
shared_nameAdam/dense/kernel/m
|
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m*
_output_shapes
:	?$@*
dtype0
z
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameAdam/dense/bias/m
s
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes
:@*
dtype0
?
Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*&
shared_nameAdam/dense_1/kernel/m

)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m*
_output_shapes

:@*
dtype0
~
Adam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_1/bias/m
w
'Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d/kernel/m
?
(Adam/conv2d/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/m*&
_output_shapes
:*
dtype0
|
Adam/conv2d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/conv2d/bias/m
u
&Adam/conv2d/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/m*
_output_shapes
:*
dtype0
?
 Adam/batch_normalization/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/batch_normalization/gamma/m
?
4Adam/batch_normalization/gamma/m/Read/ReadVariableOpReadVariableOp Adam/batch_normalization/gamma/m*
_output_shapes
:*
dtype0
?
Adam/batch_normalization/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/batch_normalization/beta/m
?
3Adam/batch_normalization/beta/m/Read/ReadVariableOpReadVariableOpAdam/batch_normalization/beta/m*
_output_shapes
:*
dtype0
?
Adam/conv2d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*'
shared_nameAdam/conv2d_1/kernel/m
?
*Adam/conv2d_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/m*&
_output_shapes
:	*
dtype0
?
Adam/conv2d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*%
shared_nameAdam/conv2d_1/bias/m
y
(Adam/conv2d_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/m*
_output_shapes
:	*
dtype0
?
Adam/conv2d/kernel/m_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d/kernel/m_1
?
*Adam/conv2d/kernel/m_1/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/m_1*&
_output_shapes
:*
dtype0
?
Adam/conv2d/bias/m_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d/bias/m_1
y
(Adam/conv2d/bias/m_1/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/m_1*
_output_shapes
:*
dtype0
?
"Adam/batch_normalization/gamma/m_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization/gamma/m_1
?
6Adam/batch_normalization/gamma/m_1/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization/gamma/m_1*
_output_shapes
:*
dtype0
?
!Adam/batch_normalization/beta/m_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization/beta/m_1
?
5Adam/batch_normalization/beta/m_1/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization/beta/m_1*
_output_shapes
:*
dtype0
?
Adam/conv2d_1/kernel/m_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_1/kernel/m_1
?
,Adam/conv2d_1/kernel/m_1/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/m_1*&
_output_shapes
:*
dtype0
?
Adam/conv2d_1/bias/m_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_1/bias/m_1
}
*Adam/conv2d_1/bias/m_1/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/m_1*
_output_shapes
:*
dtype0
?
Adam/conv2d/kernel/m_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d/kernel/m_2
?
*Adam/conv2d/kernel/m_2/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/m_2*&
_output_shapes
:*
dtype0
?
Adam/conv2d/bias/m_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d/bias/m_2
y
(Adam/conv2d/bias/m_2/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/m_2*
_output_shapes
:*
dtype0
?
"Adam/batch_normalization/gamma/m_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization/gamma/m_2
?
6Adam/batch_normalization/gamma/m_2/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization/gamma/m_2*
_output_shapes
:*
dtype0
?
!Adam/batch_normalization/beta/m_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization/beta/m_2
?
5Adam/batch_normalization/beta/m_2/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization/beta/m_2*
_output_shapes
:*
dtype0
?
Adam/conv2d_1/kernel/m_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_1/kernel/m_2
?
,Adam/conv2d_1/kernel/m_2/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/m_2*&
_output_shapes
:*
dtype0
?
Adam/conv2d_1/bias/m_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_1/bias/m_2
}
*Adam/conv2d_1/bias/m_2/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/m_2*
_output_shapes
:*
dtype0
?
Adam/conv2d/kernel/m_3VarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d/kernel/m_3
?
*Adam/conv2d/kernel/m_3/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/m_3*&
_output_shapes
:*
dtype0
?
Adam/conv2d/bias/m_3VarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d/bias/m_3
y
(Adam/conv2d/bias/m_3/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/m_3*
_output_shapes
:*
dtype0
?
"Adam/batch_normalization/gamma/m_3VarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization/gamma/m_3
?
6Adam/batch_normalization/gamma/m_3/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization/gamma/m_3*
_output_shapes
:*
dtype0
?
!Adam/batch_normalization/beta/m_3VarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization/beta/m_3
?
5Adam/batch_normalization/beta/m_3/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization/beta/m_3*
_output_shapes
:*
dtype0
?
Adam/conv2d_1/kernel/m_3VarHandleOp*
_output_shapes
: *
dtype0*
shape:	*)
shared_nameAdam/conv2d_1/kernel/m_3
?
,Adam/conv2d_1/kernel/m_3/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/m_3*&
_output_shapes
:	*
dtype0
?
Adam/conv2d_1/bias/m_3VarHandleOp*
_output_shapes
: *
dtype0*
shape:	*'
shared_nameAdam/conv2d_1/bias/m_3
}
*Adam/conv2d_1/bias/m_3/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/m_3*
_output_shapes
:	*
dtype0
?
Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?$@*$
shared_nameAdam/dense/kernel/v
|
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v*
_output_shapes
:	?$@*
dtype0
z
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_nameAdam/dense/bias/v
s
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes
:@*
dtype0
?
Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*&
shared_nameAdam/dense_1/kernel/v

)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v*
_output_shapes

:@*
dtype0
~
Adam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_1/bias/v
w
'Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv2d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d/kernel/v
?
(Adam/conv2d/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/v*&
_output_shapes
:*
dtype0
|
Adam/conv2d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/conv2d/bias/v
u
&Adam/conv2d/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/v*
_output_shapes
:*
dtype0
?
 Adam/batch_normalization/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/batch_normalization/gamma/v
?
4Adam/batch_normalization/gamma/v/Read/ReadVariableOpReadVariableOp Adam/batch_normalization/gamma/v*
_output_shapes
:*
dtype0
?
Adam/batch_normalization/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*0
shared_name!Adam/batch_normalization/beta/v
?
3Adam/batch_normalization/beta/v/Read/ReadVariableOpReadVariableOpAdam/batch_normalization/beta/v*
_output_shapes
:*
dtype0
?
Adam/conv2d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*'
shared_nameAdam/conv2d_1/kernel/v
?
*Adam/conv2d_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/v*&
_output_shapes
:	*
dtype0
?
Adam/conv2d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*%
shared_nameAdam/conv2d_1/bias/v
y
(Adam/conv2d_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/v*
_output_shapes
:	*
dtype0
?
Adam/conv2d/kernel/v_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d/kernel/v_1
?
*Adam/conv2d/kernel/v_1/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/v_1*&
_output_shapes
:*
dtype0
?
Adam/conv2d/bias/v_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d/bias/v_1
y
(Adam/conv2d/bias/v_1/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/v_1*
_output_shapes
:*
dtype0
?
"Adam/batch_normalization/gamma/v_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization/gamma/v_1
?
6Adam/batch_normalization/gamma/v_1/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization/gamma/v_1*
_output_shapes
:*
dtype0
?
!Adam/batch_normalization/beta/v_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization/beta/v_1
?
5Adam/batch_normalization/beta/v_1/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization/beta/v_1*
_output_shapes
:*
dtype0
?
Adam/conv2d_1/kernel/v_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_1/kernel/v_1
?
,Adam/conv2d_1/kernel/v_1/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/v_1*&
_output_shapes
:*
dtype0
?
Adam/conv2d_1/bias/v_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_1/bias/v_1
}
*Adam/conv2d_1/bias/v_1/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/v_1*
_output_shapes
:*
dtype0
?
Adam/conv2d/kernel/v_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d/kernel/v_2
?
*Adam/conv2d/kernel/v_2/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/v_2*&
_output_shapes
:*
dtype0
?
Adam/conv2d/bias/v_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d/bias/v_2
y
(Adam/conv2d/bias/v_2/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/v_2*
_output_shapes
:*
dtype0
?
"Adam/batch_normalization/gamma/v_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization/gamma/v_2
?
6Adam/batch_normalization/gamma/v_2/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization/gamma/v_2*
_output_shapes
:*
dtype0
?
!Adam/batch_normalization/beta/v_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization/beta/v_2
?
5Adam/batch_normalization/beta/v_2/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization/beta/v_2*
_output_shapes
:*
dtype0
?
Adam/conv2d_1/kernel/v_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameAdam/conv2d_1/kernel/v_2
?
,Adam/conv2d_1/kernel/v_2/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/v_2*&
_output_shapes
:*
dtype0
?
Adam/conv2d_1/bias/v_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_1/bias/v_2
}
*Adam/conv2d_1/bias/v_2/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/v_2*
_output_shapes
:*
dtype0
?
Adam/conv2d/kernel/v_3VarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d/kernel/v_3
?
*Adam/conv2d/kernel/v_3/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/v_3*&
_output_shapes
:*
dtype0
?
Adam/conv2d/bias/v_3VarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d/bias/v_3
y
(Adam/conv2d/bias/v_3/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/v_3*
_output_shapes
:*
dtype0
?
"Adam/batch_normalization/gamma/v_3VarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization/gamma/v_3
?
6Adam/batch_normalization/gamma/v_3/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization/gamma/v_3*
_output_shapes
:*
dtype0
?
!Adam/batch_normalization/beta/v_3VarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/batch_normalization/beta/v_3
?
5Adam/batch_normalization/beta/v_3/Read/ReadVariableOpReadVariableOp!Adam/batch_normalization/beta/v_3*
_output_shapes
:*
dtype0
?
Adam/conv2d_1/kernel/v_3VarHandleOp*
_output_shapes
: *
dtype0*
shape:	*)
shared_nameAdam/conv2d_1/kernel/v_3
?
,Adam/conv2d_1/kernel/v_3/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/v_3*&
_output_shapes
:	*
dtype0
?
Adam/conv2d_1/bias/v_3VarHandleOp*
_output_shapes
: *
dtype0*
shape:	*'
shared_nameAdam/conv2d_1/bias/v_3
}
*Adam/conv2d_1/bias/v_3/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/v_3*
_output_shapes
:	*
dtype0

NoOpNoOp
??
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*??
value??B?? B??
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
layer_with_weights-1
layer-4
layer_with_weights-2
layer-5
layer-6
layer-7
	layer-8

layer-9
layer_with_weights-3
layer-10
layer-11
layer-12
layer_with_weights-4
layer-13
layer_with_weights-5
layer-14
	optimizer
trainable_variables
regularization_losses
	variables
	keras_api

signatures
 
?

kernel_gen
kernel_reshape
input_patches_reshape
output_reshape
trainable_variables
regularization_losses
	variables
	keras_api
R
trainable_variables
regularization_losses
 	variables
!	keras_api
R
"trainable_variables
#regularization_losses
$	variables
%	keras_api
?
&stride_layer
'
kernel_gen
(kernel_reshape
)input_patches_reshape
*output_reshape
+trainable_variables
,regularization_losses
-	variables
.	keras_api
?
/stride_layer
0
kernel_gen
1kernel_reshape
2input_patches_reshape
3output_reshape
4trainable_variables
5regularization_losses
6	variables
7	keras_api
R
8trainable_variables
9regularization_losses
:	variables
;	keras_api
R
<trainable_variables
=regularization_losses
>	variables
?	keras_api
R
@trainable_variables
Aregularization_losses
B	variables
C	keras_api
R
Dtrainable_variables
Eregularization_losses
F	variables
G	keras_api
?
H
kernel_gen
Ikernel_reshape
Jinput_patches_reshape
Koutput_reshape
Ltrainable_variables
Mregularization_losses
N	variables
O	keras_api
R
Ptrainable_variables
Qregularization_losses
R	variables
S	keras_api
R
Ttrainable_variables
Uregularization_losses
V	variables
W	keras_api
h

Xkernel
Ybias
Ztrainable_variables
[regularization_losses
\	variables
]	keras_api
h

^kernel
_bias
`trainable_variables
aregularization_losses
b	variables
c	keras_api
?
diter

ebeta_1

fbeta_2
	gdecay
hlearning_rateXm?Ym?^m?_m?im?jm?km?lm?mm?nm?om?pm?qm?rm?sm?tm?um?vm?wm?xm?ym?zm?{m?|m?}m?~m?m?	?m?Xv?Yv?^v?_v?iv?jv?kv?lv?mv?nv?ov?pv?qv?rv?sv?tv?uv?vv?wv?xv?yv?zv?{v?|v?}v?~v?v?	?v?
?
i0
j1
k2
l3
m4
n5
o6
p7
q8
r9
s10
t11
u12
v13
w14
x15
y16
z17
{18
|19
}20
~21
22
?23
X24
Y25
^26
_27
 
?
i0
j1
k2
l3
m4
n5
?6
?7
o8
p9
q10
r11
s12
t13
?14
?15
u16
v17
w18
x19
y20
z21
?22
?23
{24
|25
}26
~27
28
?29
?30
?31
X32
Y33
^34
_35
?
trainable_variables
?layer_metrics
regularization_losses
?metrics
 ?layer_regularization_losses
?layers
	variables
?non_trainable_variables
 
?
?layer_with_weights-0
?layer-0
?layer_with_weights-1
?layer-1
?layer-2
?layer_with_weights-2
?layer-3
?trainable_variables
?regularization_losses
?	variables
?	keras_api
V
?trainable_variables
?regularization_losses
?	variables
?	keras_api
V
?trainable_variables
?regularization_losses
?	variables
?	keras_api
V
?trainable_variables
?regularization_losses
?	variables
?	keras_api
*
i0
j1
k2
l3
m4
n5
 
:
i0
j1
k2
l3
m4
n5
?6
?7
?
trainable_variables
?layer_metrics
regularization_losses
?metrics
 ?layer_regularization_losses
?layers
	variables
?non_trainable_variables
 
 
 
?
trainable_variables
?layer_metrics
regularization_losses
?metrics
 ?layer_regularization_losses
?layers
 	variables
?non_trainable_variables
 
 
 
?
"trainable_variables
?layer_metrics
#regularization_losses
?metrics
 ?layer_regularization_losses
?layers
$	variables
?non_trainable_variables
V
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?
?layer_with_weights-0
?layer-0
?layer_with_weights-1
?layer-1
?layer-2
?layer_with_weights-2
?layer-3
?trainable_variables
?regularization_losses
?	variables
?	keras_api
V
?trainable_variables
?regularization_losses
?	variables
?	keras_api
V
?trainable_variables
?regularization_losses
?	variables
?	keras_api
V
?trainable_variables
?regularization_losses
?	variables
?	keras_api
*
o0
p1
q2
r3
s4
t5
 
:
o0
p1
q2
r3
s4
t5
?6
?7
?
+trainable_variables
?layer_metrics
,regularization_losses
?metrics
 ?layer_regularization_losses
?layers
-	variables
?non_trainable_variables
V
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?
?layer_with_weights-0
?layer-0
?layer_with_weights-1
?layer-1
?layer-2
?layer_with_weights-2
?layer-3
?trainable_variables
?regularization_losses
?	variables
?	keras_api
V
?trainable_variables
?regularization_losses
?	variables
?	keras_api
V
?trainable_variables
?regularization_losses
?	variables
?	keras_api
V
?trainable_variables
?regularization_losses
?	variables
?	keras_api
*
u0
v1
w2
x3
y4
z5
 
:
u0
v1
w2
x3
y4
z5
?6
?7
?
4trainable_variables
?layer_metrics
5regularization_losses
?metrics
 ?layer_regularization_losses
?layers
6	variables
?non_trainable_variables
 
 
 
?
8trainable_variables
?layer_metrics
9regularization_losses
?metrics
 ?layer_regularization_losses
?layers
:	variables
?non_trainable_variables
 
 
 
?
<trainable_variables
?layer_metrics
=regularization_losses
?metrics
 ?layer_regularization_losses
?layers
>	variables
?non_trainable_variables
 
 
 
?
@trainable_variables
?layer_metrics
Aregularization_losses
?metrics
 ?layer_regularization_losses
?layers
B	variables
?non_trainable_variables
 
 
 
?
Dtrainable_variables
?layer_metrics
Eregularization_losses
?metrics
 ?layer_regularization_losses
?layers
F	variables
?non_trainable_variables
?
?layer_with_weights-0
?layer-0
?layer_with_weights-1
?layer-1
?layer-2
?layer_with_weights-2
?layer-3
?trainable_variables
?regularization_losses
?	variables
?	keras_api
V
?trainable_variables
?regularization_losses
?	variables
?	keras_api
V
?trainable_variables
?regularization_losses
?	variables
?	keras_api
V
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+
{0
|1
}2
~3
4
?5
 
;
{0
|1
}2
~3
4
?5
?6
?7
?
Ltrainable_variables
?layer_metrics
Mregularization_losses
?metrics
 ?layer_regularization_losses
?layers
N	variables
?non_trainable_variables
 
 
 
?
Ptrainable_variables
?layer_metrics
Qregularization_losses
?metrics
 ?layer_regularization_losses
?layers
R	variables
?non_trainable_variables
 
 
 
?
Ttrainable_variables
?layer_metrics
Uregularization_losses
?metrics
 ?layer_regularization_losses
?layers
V	variables
?non_trainable_variables
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

X0
Y1
 

X0
Y1
?
Ztrainable_variables
?layer_metrics
[regularization_losses
?metrics
 ?layer_regularization_losses
?layers
\	variables
?non_trainable_variables
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

^0
_1
 

^0
_1
?
`trainable_variables
?layer_metrics
aregularization_losses
?metrics
 ?layer_regularization_losses
?layers
b	variables
?non_trainable_variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEconv2d/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEconv2d/bias0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEbatch_normalization/gamma0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEbatch_normalization/beta0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv2d_1/kernel0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEconv2d_1/bias0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv2d/kernel_10trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEconv2d/bias_10trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEbatch_normalization/gamma_10trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEbatch_normalization/beta_10trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_1/kernel_11trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEconv2d_1/bias_11trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEconv2d/kernel_21trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEconv2d/bias_21trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEbatch_normalization/gamma_21trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEbatch_normalization/beta_21trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_1/kernel_21trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEconv2d_1/bias_21trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEconv2d/kernel_31trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEconv2d/bias_31trainable_variables/19/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEbatch_normalization/gamma_31trainable_variables/20/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEbatch_normalization/beta_31trainable_variables/21/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_1/kernel_31trainable_variables/22/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEconv2d_1/bias_31trainable_variables/23/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEbatch_normalization/moving_mean&variables/6/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUE#batch_normalization/moving_variance&variables/7/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE!batch_normalization/moving_mean_1'variables/14/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE%batch_normalization/moving_variance_1'variables/15/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE!batch_normalization/moving_mean_2'variables/22/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE%batch_normalization/moving_variance_2'variables/23/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUE!batch_normalization/moving_mean_3'variables/30/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUE%batch_normalization/moving_variance_3'variables/31/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1
 
n
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
@
?0
?1
?2
?3
?4
?5
?6
?7
l

ikernel
jbias
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?
	?axis
	kgamma
lbeta
?moving_mean
?moving_variance
?trainable_variables
?regularization_losses
?	variables
?	keras_api
V
?trainable_variables
?regularization_losses
?	variables
?	keras_api
l

mkernel
nbias
?trainable_variables
?regularization_losses
?	variables
?	keras_api
*
i0
j1
k2
l3
m4
n5
 
:
i0
j1
k2
l3
?4
?5
m6
n7
?
?trainable_variables
?layer_metrics
?regularization_losses
?metrics
 ?layer_regularization_losses
?layers
?	variables
?non_trainable_variables
 
 
 
?
?trainable_variables
?layer_metrics
?regularization_losses
?metrics
 ?layer_regularization_losses
?layers
?	variables
?non_trainable_variables
 
 
 
?
?trainable_variables
?layer_metrics
?regularization_losses
?metrics
 ?layer_regularization_losses
?layers
?	variables
?non_trainable_variables
 
 
 
?
?trainable_variables
?layer_metrics
?regularization_losses
?metrics
 ?layer_regularization_losses
?layers
?	variables
?non_trainable_variables
 
 
 

0
1
2
3

?0
?1
 
 
 
 
 
 
 
 
 
 
 
 
 
?
?trainable_variables
?layer_metrics
?regularization_losses
?metrics
 ?layer_regularization_losses
?layers
?	variables
?non_trainable_variables
l

okernel
pbias
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?
	?axis
	qgamma
rbeta
?moving_mean
?moving_variance
?trainable_variables
?regularization_losses
?	variables
?	keras_api
V
?trainable_variables
?regularization_losses
?	variables
?	keras_api
l

skernel
tbias
?trainable_variables
?regularization_losses
?	variables
?	keras_api
*
o0
p1
q2
r3
s4
t5
 
:
o0
p1
q2
r3
?4
?5
s6
t7
?
?trainable_variables
?layer_metrics
?regularization_losses
?metrics
 ?layer_regularization_losses
?layers
?	variables
?non_trainable_variables
 
 
 
?
?trainable_variables
?layer_metrics
?regularization_losses
?metrics
 ?layer_regularization_losses
?layers
?	variables
?non_trainable_variables
 
 
 
?
?trainable_variables
?layer_metrics
?regularization_losses
?metrics
 ?layer_regularization_losses
?layers
?	variables
?non_trainable_variables
 
 
 
?
?trainable_variables
?layer_metrics
?regularization_losses
?metrics
 ?layer_regularization_losses
?layers
?	variables
?non_trainable_variables
 
 
 
#
&0
'1
(2
)3
*4

?0
?1
 
 
 
?
?trainable_variables
?layer_metrics
?regularization_losses
?metrics
 ?layer_regularization_losses
?layers
?	variables
?non_trainable_variables
l

ukernel
vbias
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?
	?axis
	wgamma
xbeta
?moving_mean
?moving_variance
?trainable_variables
?regularization_losses
?	variables
?	keras_api
V
?trainable_variables
?regularization_losses
?	variables
?	keras_api
l

ykernel
zbias
?trainable_variables
?regularization_losses
?	variables
?	keras_api
*
u0
v1
w2
x3
y4
z5
 
:
u0
v1
w2
x3
?4
?5
y6
z7
?
?trainable_variables
?layer_metrics
?regularization_losses
?metrics
 ?layer_regularization_losses
?layers
?	variables
?non_trainable_variables
 
 
 
?
?trainable_variables
?layer_metrics
?regularization_losses
?metrics
 ?layer_regularization_losses
?layers
?	variables
?non_trainable_variables
 
 
 
?
?trainable_variables
?layer_metrics
?regularization_losses
?metrics
 ?layer_regularization_losses
?layers
?	variables
?non_trainable_variables
 
 
 
?
?trainable_variables
?layer_metrics
?regularization_losses
?metrics
 ?layer_regularization_losses
?layers
?	variables
?non_trainable_variables
 
 
 
#
/0
01
12
23
34

?0
?1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
l

{kernel
|bias
?trainable_variables
?regularization_losses
?	variables
?	keras_api
?
	?axis
	}gamma
~beta
?moving_mean
?moving_variance
?trainable_variables
?regularization_losses
?	variables
?	keras_api
V
?trainable_variables
?regularization_losses
?	variables
?	keras_api
m

kernel
	?bias
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+
{0
|1
}2
~3
4
?5
 
;
{0
|1
}2
~3
?4
?5
6
?7
?
?trainable_variables
?layer_metrics
?regularization_losses
?metrics
 ?layer_regularization_losses
?layers
?	variables
?non_trainable_variables
 
 
 
?
?trainable_variables
?layer_metrics
?regularization_losses
?metrics
 ?layer_regularization_losses
?layers
?	variables
?non_trainable_variables
 
 
 
?
?trainable_variables
?layer_metrics
?regularization_losses
?metrics
 ?layer_regularization_losses
?layers
?	variables
?non_trainable_variables
 
 
 
?
?trainable_variables
?layer_metrics
?regularization_losses
?metrics
 ?layer_regularization_losses
?layers
?	variables
?non_trainable_variables
 
 
 

H0
I1
J2
K3

?0
?1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

?total

?count
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api

i0
j1
 

i0
j1
?
?trainable_variables
?layer_metrics
?regularization_losses
?metrics
 ?layer_regularization_losses
?layers
?	variables
?non_trainable_variables
 

k0
l1
 

k0
l1
?2
?3
?
?trainable_variables
?layer_metrics
?regularization_losses
?metrics
 ?layer_regularization_losses
?layers
?	variables
?non_trainable_variables
 
 
 
?
?trainable_variables
?layer_metrics
?regularization_losses
?metrics
 ?layer_regularization_losses
?layers
?	variables
?non_trainable_variables

m0
n1
 

m0
n1
?
?trainable_variables
?layer_metrics
?regularization_losses
?metrics
 ?layer_regularization_losses
?layers
?	variables
?non_trainable_variables
 
 
 
 
?0
?1
?2
?3

?0
?1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

o0
p1
 

o0
p1
?
?trainable_variables
?layer_metrics
?regularization_losses
?metrics
 ?layer_regularization_losses
?layers
?	variables
?non_trainable_variables
 

q0
r1
 

q0
r1
?2
?3
?
?trainable_variables
?layer_metrics
?regularization_losses
?metrics
 ?layer_regularization_losses
?layers
?	variables
?non_trainable_variables
 
 
 
?
?trainable_variables
?layer_metrics
?regularization_losses
?metrics
 ?layer_regularization_losses
?layers
?	variables
?non_trainable_variables

s0
t1
 

s0
t1
?
?trainable_variables
?layer_metrics
?regularization_losses
?metrics
 ?layer_regularization_losses
?layers
?	variables
?non_trainable_variables
 
 
 
 
?0
?1
?2
?3

?0
?1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

u0
v1
 

u0
v1
?
?trainable_variables
?layer_metrics
?regularization_losses
?metrics
 ?layer_regularization_losses
?layers
?	variables
?non_trainable_variables
 

w0
x1
 

w0
x1
?2
?3
?
?trainable_variables
?layer_metrics
?regularization_losses
?metrics
 ?layer_regularization_losses
?layers
?	variables
?non_trainable_variables
 
 
 
?
?trainable_variables
?layer_metrics
?regularization_losses
?metrics
 ?layer_regularization_losses
?layers
?	variables
?non_trainable_variables

y0
z1
 

y0
z1
?
?trainable_variables
?layer_metrics
?regularization_losses
?metrics
 ?layer_regularization_losses
?layers
?	variables
?non_trainable_variables
 
 
 
 
?0
?1
?2
?3

?0
?1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

{0
|1
 

{0
|1
?
?trainable_variables
?layer_metrics
?regularization_losses
?metrics
 ?layer_regularization_losses
?layers
?	variables
?non_trainable_variables
 

}0
~1
 

}0
~1
?2
?3
?
?trainable_variables
?layer_metrics
?regularization_losses
?metrics
 ?layer_regularization_losses
?layers
?	variables
?non_trainable_variables
 
 
 
?
?trainable_variables
?layer_metrics
?regularization_losses
?metrics
 ?layer_regularization_losses
?layers
?	variables
?non_trainable_variables

0
?1
 

0
?1
?
?trainable_variables
?layer_metrics
?regularization_losses
?metrics
 ?layer_regularization_losses
?layers
?	variables
?non_trainable_variables
 
 
 
 
?0
?1
?2
?3

?0
?1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
 
 
 
 
 
 
 
 
 

?0
?1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

?0
?1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

?0
?1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

?0
?1
 
 
 
 
 
 
 
 
 
 
{y
VARIABLE_VALUEAdam/dense/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_1/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/conv2d/kernel/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/conv2d/bias/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adam/batch_normalization/gamma/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/batch_normalization/beta/mLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d_1/kernel/mLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/conv2d_1/bias/mLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d/kernel/m_1Ltrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/conv2d/bias/m_1Ltrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization/gamma/m_1Ltrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adam/batch_normalization/beta/m_1Ltrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_1/kernel/m_1Mtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/conv2d_1/bias/m_1Mtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/conv2d/kernel/m_2Mtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv2d/bias/m_2Mtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization/gamma/m_2Mtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adam/batch_normalization/beta/m_2Mtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_1/kernel/m_2Mtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/conv2d_1/bias/m_2Mtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/conv2d/kernel/m_3Mtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv2d/bias/m_3Mtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization/gamma/m_3Mtrainable_variables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adam/batch_normalization/beta/m_3Mtrainable_variables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_1/kernel/m_3Mtrainable_variables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/conv2d_1/bias/m_3Mtrainable_variables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_1/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/conv2d/kernel/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/conv2d/bias/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE Adam/batch_normalization/gamma/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/batch_normalization/beta/vLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d_1/kernel/vLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/conv2d_1/bias/vLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/conv2d/kernel/v_1Ltrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/conv2d/bias/v_1Ltrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization/gamma/v_1Ltrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adam/batch_normalization/beta/v_1Ltrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_1/kernel/v_1Mtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/conv2d_1/bias/v_1Mtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/conv2d/kernel/v_2Mtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv2d/bias/v_2Mtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization/gamma/v_2Mtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adam/batch_normalization/beta/v_2Mtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_1/kernel/v_2Mtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/conv2d_1/bias/v_2Mtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/conv2d/kernel/v_3Mtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/conv2d/bias/v_3Mtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE"Adam/batch_normalization/gamma/v_3Mtrainable_variables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE!Adam/batch_normalization/beta/v_3Mtrainable_variables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_1/kernel/v_3Mtrainable_variables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/conv2d_1/bias/v_3Mtrainable_variables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_1Placeholder*1
_output_shapes
:???????????*
dtype0*&
shape:???????????
?	
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv2d/kernelconv2d/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_varianceconv2d_1/kernelconv2d_1/biasconv2d/kernel_2conv2d/bias_2batch_normalization/gamma_2batch_normalization/beta_2!batch_normalization/moving_mean_2%batch_normalization/moving_variance_2conv2d_1/kernel_2conv2d_1/bias_2conv2d/kernel_1conv2d/bias_1batch_normalization/gamma_1batch_normalization/beta_1!batch_normalization/moving_mean_1%batch_normalization/moving_variance_1conv2d_1/kernel_1conv2d_1/bias_1conv2d/kernel_3conv2d/bias_3batch_normalization/gamma_3batch_normalization/beta_3!batch_normalization/moving_mean_3%batch_normalization/moving_variance_3conv2d_1/kernel_3conv2d_1/bias_3dense/kernel
dense/biasdense_1/kerneldense_1/bias*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*F
_read_only_resource_inputs(
&$	
 !"#$*0
config_proto 

CPU

GPU2*0J 8? *-
f(R&
$__inference_signature_wrapper_305262
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?%
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp-batch_normalization/gamma/Read/ReadVariableOp,batch_normalization/beta/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp#conv2d/kernel_1/Read/ReadVariableOp!conv2d/bias_1/Read/ReadVariableOp/batch_normalization/gamma_1/Read/ReadVariableOp.batch_normalization/beta_1/Read/ReadVariableOp%conv2d_1/kernel_1/Read/ReadVariableOp#conv2d_1/bias_1/Read/ReadVariableOp#conv2d/kernel_2/Read/ReadVariableOp!conv2d/bias_2/Read/ReadVariableOp/batch_normalization/gamma_2/Read/ReadVariableOp.batch_normalization/beta_2/Read/ReadVariableOp%conv2d_1/kernel_2/Read/ReadVariableOp#conv2d_1/bias_2/Read/ReadVariableOp#conv2d/kernel_3/Read/ReadVariableOp!conv2d/bias_3/Read/ReadVariableOp/batch_normalization/gamma_3/Read/ReadVariableOp.batch_normalization/beta_3/Read/ReadVariableOp%conv2d_1/kernel_3/Read/ReadVariableOp#conv2d_1/bias_3/Read/ReadVariableOp3batch_normalization/moving_mean/Read/ReadVariableOp7batch_normalization/moving_variance/Read/ReadVariableOp5batch_normalization/moving_mean_1/Read/ReadVariableOp9batch_normalization/moving_variance_1/Read/ReadVariableOp5batch_normalization/moving_mean_2/Read/ReadVariableOp9batch_normalization/moving_variance_2/Read/ReadVariableOp5batch_normalization/moving_mean_3/Read/ReadVariableOp9batch_normalization/moving_variance_3/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp)Adam/dense_1/kernel/m/Read/ReadVariableOp'Adam/dense_1/bias/m/Read/ReadVariableOp(Adam/conv2d/kernel/m/Read/ReadVariableOp&Adam/conv2d/bias/m/Read/ReadVariableOp4Adam/batch_normalization/gamma/m/Read/ReadVariableOp3Adam/batch_normalization/beta/m/Read/ReadVariableOp*Adam/conv2d_1/kernel/m/Read/ReadVariableOp(Adam/conv2d_1/bias/m/Read/ReadVariableOp*Adam/conv2d/kernel/m_1/Read/ReadVariableOp(Adam/conv2d/bias/m_1/Read/ReadVariableOp6Adam/batch_normalization/gamma/m_1/Read/ReadVariableOp5Adam/batch_normalization/beta/m_1/Read/ReadVariableOp,Adam/conv2d_1/kernel/m_1/Read/ReadVariableOp*Adam/conv2d_1/bias/m_1/Read/ReadVariableOp*Adam/conv2d/kernel/m_2/Read/ReadVariableOp(Adam/conv2d/bias/m_2/Read/ReadVariableOp6Adam/batch_normalization/gamma/m_2/Read/ReadVariableOp5Adam/batch_normalization/beta/m_2/Read/ReadVariableOp,Adam/conv2d_1/kernel/m_2/Read/ReadVariableOp*Adam/conv2d_1/bias/m_2/Read/ReadVariableOp*Adam/conv2d/kernel/m_3/Read/ReadVariableOp(Adam/conv2d/bias/m_3/Read/ReadVariableOp6Adam/batch_normalization/gamma/m_3/Read/ReadVariableOp5Adam/batch_normalization/beta/m_3/Read/ReadVariableOp,Adam/conv2d_1/kernel/m_3/Read/ReadVariableOp*Adam/conv2d_1/bias/m_3/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOp)Adam/dense_1/kernel/v/Read/ReadVariableOp'Adam/dense_1/bias/v/Read/ReadVariableOp(Adam/conv2d/kernel/v/Read/ReadVariableOp&Adam/conv2d/bias/v/Read/ReadVariableOp4Adam/batch_normalization/gamma/v/Read/ReadVariableOp3Adam/batch_normalization/beta/v/Read/ReadVariableOp*Adam/conv2d_1/kernel/v/Read/ReadVariableOp(Adam/conv2d_1/bias/v/Read/ReadVariableOp*Adam/conv2d/kernel/v_1/Read/ReadVariableOp(Adam/conv2d/bias/v_1/Read/ReadVariableOp6Adam/batch_normalization/gamma/v_1/Read/ReadVariableOp5Adam/batch_normalization/beta/v_1/Read/ReadVariableOp,Adam/conv2d_1/kernel/v_1/Read/ReadVariableOp*Adam/conv2d_1/bias/v_1/Read/ReadVariableOp*Adam/conv2d/kernel/v_2/Read/ReadVariableOp(Adam/conv2d/bias/v_2/Read/ReadVariableOp6Adam/batch_normalization/gamma/v_2/Read/ReadVariableOp5Adam/batch_normalization/beta/v_2/Read/ReadVariableOp,Adam/conv2d_1/kernel/v_2/Read/ReadVariableOp*Adam/conv2d_1/bias/v_2/Read/ReadVariableOp*Adam/conv2d/kernel/v_3/Read/ReadVariableOp(Adam/conv2d/bias/v_3/Read/ReadVariableOp6Adam/batch_normalization/gamma/v_3/Read/ReadVariableOp5Adam/batch_normalization/beta/v_3/Read/ReadVariableOp,Adam/conv2d_1/kernel/v_3/Read/ReadVariableOp*Adam/conv2d_1/bias/v_3/Read/ReadVariableOpConst*r
Tink
i2g	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *(
f#R!
__inference__traced_save_308322
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense/kernel
dense/biasdense_1/kerneldense_1/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rateconv2d/kernelconv2d/biasbatch_normalization/gammabatch_normalization/betaconv2d_1/kernelconv2d_1/biasconv2d/kernel_1conv2d/bias_1batch_normalization/gamma_1batch_normalization/beta_1conv2d_1/kernel_1conv2d_1/bias_1conv2d/kernel_2conv2d/bias_2batch_normalization/gamma_2batch_normalization/beta_2conv2d_1/kernel_2conv2d_1/bias_2conv2d/kernel_3conv2d/bias_3batch_normalization/gamma_3batch_normalization/beta_3conv2d_1/kernel_3conv2d_1/bias_3batch_normalization/moving_mean#batch_normalization/moving_variance!batch_normalization/moving_mean_1%batch_normalization/moving_variance_1!batch_normalization/moving_mean_2%batch_normalization/moving_variance_2!batch_normalization/moving_mean_3%batch_normalization/moving_variance_3totalcounttotal_1count_1Adam/dense/kernel/mAdam/dense/bias/mAdam/dense_1/kernel/mAdam/dense_1/bias/mAdam/conv2d/kernel/mAdam/conv2d/bias/m Adam/batch_normalization/gamma/mAdam/batch_normalization/beta/mAdam/conv2d_1/kernel/mAdam/conv2d_1/bias/mAdam/conv2d/kernel/m_1Adam/conv2d/bias/m_1"Adam/batch_normalization/gamma/m_1!Adam/batch_normalization/beta/m_1Adam/conv2d_1/kernel/m_1Adam/conv2d_1/bias/m_1Adam/conv2d/kernel/m_2Adam/conv2d/bias/m_2"Adam/batch_normalization/gamma/m_2!Adam/batch_normalization/beta/m_2Adam/conv2d_1/kernel/m_2Adam/conv2d_1/bias/m_2Adam/conv2d/kernel/m_3Adam/conv2d/bias/m_3"Adam/batch_normalization/gamma/m_3!Adam/batch_normalization/beta/m_3Adam/conv2d_1/kernel/m_3Adam/conv2d_1/bias/m_3Adam/dense/kernel/vAdam/dense/bias/vAdam/dense_1/kernel/vAdam/dense_1/bias/vAdam/conv2d/kernel/vAdam/conv2d/bias/v Adam/batch_normalization/gamma/vAdam/batch_normalization/beta/vAdam/conv2d_1/kernel/vAdam/conv2d_1/bias/vAdam/conv2d/kernel/v_1Adam/conv2d/bias/v_1"Adam/batch_normalization/gamma/v_1!Adam/batch_normalization/beta/v_1Adam/conv2d_1/kernel/v_1Adam/conv2d_1/bias/v_1Adam/conv2d/kernel/v_2Adam/conv2d/bias/v_2"Adam/batch_normalization/gamma/v_2!Adam/batch_normalization/beta/v_2Adam/conv2d_1/kernel/v_2Adam/conv2d_1/bias/v_2Adam/conv2d/kernel/v_3Adam/conv2d/bias/v_3"Adam/batch_normalization/gamma/v_3!Adam/batch_normalization/beta/v_3Adam/conv2d_1/kernel/v_3Adam/conv2d_1/bias/v_3*q
Tinj
h2f*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *+
f&R$
"__inference__traced_restore_308635??1
?
?
O__inference_batch_normalization_layer_call_and_return_conditional_losses_307879

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
D
(__inference_re_lu_3_layer_call_fn_306821

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_re_lu_3_layer_call_and_return_conditional_losses_3040372
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
+__inference_sequential_layer_call_fn_303583
conv2d_input!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5:	
	unknown_6:	
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????	*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_3035432
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????	2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:?????????
&
_user_specified_nameconv2d_input
?
B
&__inference_re_lu_layer_call_fn_307461

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_re_lu_layer_call_and_return_conditional_losses_3020282
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
J
.__inference_max_pooling2d_layer_call_fn_306212

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????pp* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_3037372
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????pp2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
B
&__inference_re_lu_layer_call_fn_307977

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_re_lu_layer_call_and_return_conditional_losses_3033882
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
4__inference_batch_normalization_layer_call_fn_307782

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????88*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_3029272
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????882

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????88: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????88
 
_user_specified_nameinputs
?
L
0__inference_max_pooling2d_1_layer_call_fn_306629

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_3039422
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????88:W S
/
_output_shapes
:?????????88
 
_user_specified_nameinputs
?
i
M__inference_average_pooling2d_layer_call_and_return_conditional_losses_302748

inputs
identity?
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingSAME*
strides
2	
AvgPool?
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_layer_call_and_return_conditional_losses_307399

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1y
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:???????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
D__inference_conv2d_1_layer_call_and_return_conditional_losses_307471

inputs8
conv2d_readvariableop_resource:	-
biasadd_readvariableop_resource:	
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????	*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????	2	
BiasAddu
IdentityIdentityBiasAdd:output:0^NoOp*
T0*1
_output_shapes
:???????????	2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
_
C__inference_re_lu_2_layer_call_and_return_conditional_losses_306591

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????882
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????882

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????88:W S
/
_output_shapes
:?????????88
 
_user_specified_nameinputs
?
?
F__inference_sequential_layer_call_and_return_conditional_losses_302961

inputs'
conv2d_302905:
conv2d_302907:(
batch_normalization_302928:(
batch_normalization_302930:(
batch_normalization_302932:(
batch_normalization_302934:)
conv2d_1_302955:
conv2d_1_302957:
identity??+batch_normalization/StatefulPartitionedCall?conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_302905conv2d_302907*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????88*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_3029042 
conv2d/StatefulPartitionedCall?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_302928batch_normalization_302930batch_normalization_302932batch_normalization_302934*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????88*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_3029272-
+batch_normalization/StatefulPartitionedCall?
re_lu/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????88* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_re_lu_layer_call_and_return_conditional_losses_3029422
re_lu/PartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallre_lu/PartitionedCall:output:0conv2d_1_302955conv2d_1_302957*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????88*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_3029542"
 conv2d_1/StatefulPartitionedCall?
IdentityIdentity)conv2d_1/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????882

Identity?
NoOpNoOp,^batch_normalization/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????88: : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall:W S
/
_output_shapes
:?????????88
 
_user_specified_nameinputs
?
?
F__inference_sequential_layer_call_and_return_conditional_losses_303097

inputs'
conv2d_303076:
conv2d_303078:(
batch_normalization_303081:(
batch_normalization_303083:(
batch_normalization_303085:(
batch_normalization_303087:)
conv2d_1_303091:
conv2d_1_303093:
identity??+batch_normalization/StatefulPartitionedCall?conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_303076conv2d_303078*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????88*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_3029042 
conv2d/StatefulPartitionedCall?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_303081batch_normalization_303083batch_normalization_303085batch_normalization_303087*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????88*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_3030292-
+batch_normalization/StatefulPartitionedCall?
re_lu/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????88* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_re_lu_layer_call_and_return_conditional_losses_3029422
re_lu/PartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallre_lu/PartitionedCall:output:0conv2d_1_303091conv2d_1_303093*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????88*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_3029542"
 conv2d_1/StatefulPartitionedCall?
IdentityIdentity)conv2d_1/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????882

Identity?
NoOpNoOp,^batch_normalization/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????88: : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall:W S
/
_output_shapes
:?????????88
 
_user_specified_nameinputs
?
?
F__inference_sequential_layer_call_and_return_conditional_losses_303543

inputs'
conv2d_303522:
conv2d_303524:(
batch_normalization_303527:(
batch_normalization_303529:(
batch_normalization_303531:(
batch_normalization_303533:)
conv2d_1_303537:	
conv2d_1_303539:	
identity??+batch_normalization/StatefulPartitionedCall?conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_303522conv2d_303524*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_3033502 
conv2d/StatefulPartitionedCall?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_303527batch_normalization_303529batch_normalization_303531batch_normalization_303533*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_3034752-
+batch_normalization/StatefulPartitionedCall?
re_lu/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_re_lu_layer_call_and_return_conditional_losses_3033882
re_lu/PartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallre_lu/PartitionedCall:output:0conv2d_1_303537conv2d_1_303539*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_3034002"
 conv2d_1/StatefulPartitionedCall?
IdentityIdentity)conv2d_1/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????	2

Identity?
NoOpNoOp,^batch_normalization/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????: : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
B
&__inference_re_lu_layer_call_fn_307805

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????88* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_re_lu_layer_call_and_return_conditional_losses_3029422
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????882

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????88:W S
/
_output_shapes
:?????????88
 
_user_specified_nameinputs
?
?
D__inference_conv2d_1_layer_call_and_return_conditional_losses_302954

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????88*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????882	
BiasAdds
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????882

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????88: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????88
 
_user_specified_nameinputs
?
?
C__inference_dense_1_layer_call_and_return_conditional_losses_306863

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Softmaxl
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?

?
+__inference_sequential_layer_call_fn_303137
conv2d_input!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????88*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_3030972
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????882

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????88: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:?????????88
&
_user_specified_nameconv2d_input
?

?
+__inference_sequential_layer_call_fn_306976

inputs!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5:	
	unknown_6:	
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????	*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_3021832
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????	2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:???????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_layer_call_and_return_conditional_losses_307517

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
i
M__inference_average_pooling2d_layer_call_and_return_conditional_losses_307095

inputs
identity?
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingSAME*
strides
2	
AvgPool?
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_layer_call_and_return_conditional_losses_303373

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?h
?	
A__inference_inv_1_layer_call_and_return_conditional_losses_304657
xJ
0sequential_conv2d_conv2d_readvariableop_resource:?
1sequential_conv2d_biasadd_readvariableop_resource:D
6sequential_batch_normalization_readvariableop_resource:F
8sequential_batch_normalization_readvariableop_1_resource:U
Gsequential_batch_normalization_fusedbatchnormv3_readvariableop_resource:W
Isequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource:L
2sequential_conv2d_1_conv2d_readvariableop_resource:	A
3sequential_conv2d_1_biasadd_readvariableop_resource:	

identity_1??-sequential/batch_normalization/AssignNewValue?/sequential/batch_normalization/AssignNewValue_1?>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp?@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?-sequential/batch_normalization/ReadVariableOp?/sequential/batch_normalization/ReadVariableOp_1?(sequential/conv2d/BiasAdd/ReadVariableOp?'sequential/conv2d/Conv2D/ReadVariableOp?*sequential/conv2d_1/BiasAdd/ReadVariableOp?)sequential/conv2d_1/Conv2D/ReadVariableOp_
IdentityIdentityx*
T0*1
_output_shapes
:???????????2

Identity?
'sequential/conv2d/Conv2D/ReadVariableOpReadVariableOp0sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02)
'sequential/conv2d/Conv2D/ReadVariableOp?
sequential/conv2d/Conv2DConv2DIdentity:output:0/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
sequential/conv2d/Conv2D?
(sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOp1sequential_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(sequential/conv2d/BiasAdd/ReadVariableOp?
sequential/conv2d/BiasAddBiasAdd!sequential/conv2d/Conv2D:output:00sequential/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
sequential/conv2d/BiasAdd?
-sequential/batch_normalization/ReadVariableOpReadVariableOp6sequential_batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential/batch_normalization/ReadVariableOp?
/sequential/batch_normalization/ReadVariableOp_1ReadVariableOp8sequential_batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype021
/sequential/batch_normalization/ReadVariableOp_1?
>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpGsequential_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02@
>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp?
@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIsequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02B
@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?
/sequential/batch_normalization/FusedBatchNormV3FusedBatchNormV3"sequential/conv2d/BiasAdd:output:05sequential/batch_normalization/ReadVariableOp:value:07sequential/batch_normalization/ReadVariableOp_1:value:0Fsequential/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Hsequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<21
/sequential/batch_normalization/FusedBatchNormV3?
-sequential/batch_normalization/AssignNewValueAssignVariableOpGsequential_batch_normalization_fusedbatchnormv3_readvariableop_resource<sequential/batch_normalization/FusedBatchNormV3:batch_mean:0?^sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02/
-sequential/batch_normalization/AssignNewValue?
/sequential/batch_normalization/AssignNewValue_1AssignVariableOpIsequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource@sequential/batch_normalization/FusedBatchNormV3:batch_variance:0A^sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype021
/sequential/batch_normalization/AssignNewValue_1?
sequential/re_lu/ReluRelu3sequential/batch_normalization/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????2
sequential/re_lu/Relu?
)sequential/conv2d_1/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:	*
dtype02+
)sequential/conv2d_1/Conv2D/ReadVariableOp?
sequential/conv2d_1/Conv2DConv2D#sequential/re_lu/Relu:activations:01sequential/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????	*
paddingVALID*
strides
2
sequential/conv2d_1/Conv2D?
*sequential/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02,
*sequential/conv2d_1/BiasAdd/ReadVariableOp?
sequential/conv2d_1/BiasAddBiasAdd#sequential/conv2d_1/Conv2D:output:02sequential/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????	2
sequential/conv2d_1/BiasAddr
reshape/ShapeShape$sequential/conv2d_1/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape/Shape?
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack?
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1?
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2?
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_sliceu
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :?2
reshape/Reshape/shape/1u
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :?2
reshape/Reshape/shape/2t
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :	2
reshape/Reshape/shape/3t
reshape/Reshape/shape/4Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/4t
reshape/Reshape/shape/5Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/5?
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0 reshape/Reshape/shape/4:output:0 reshape/Reshape/shape/5:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape?
reshape/ReshapeReshape$sequential/conv2d_1/BiasAdd:output:0reshape/Reshape/shape:output:0*
T0*9
_output_shapes'
%:#???????????	2
reshape/Reshape?
ExtractImagePatchesExtractImagePatchesx*
T0*1
_output_shapes
:???????????*
ksizes
*
paddingSAME*
rates
*
strides
2
ExtractImagePatcheso
reshape_1/ShapeShapeExtractImagePatches:patches:0*
T0*
_output_shapes
:2
reshape_1/Shape?
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_1/strided_slice/stack?
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_1?
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_2?
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_1/strided_slicey
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :?2
reshape_1/Reshape/shape/1y
reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :?2
reshape_1/Reshape/shape/2x
reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :	2
reshape_1/Reshape/shape/3x
reshape_1/Reshape/shape/4Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/4x
reshape_1/Reshape/shape/5Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/5?
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0"reshape_1/Reshape/shape/2:output:0"reshape_1/Reshape/shape/3:output:0"reshape_1/Reshape/shape/4:output:0"reshape_1/Reshape/shape/5:output:0*
N*
T0*
_output_shapes
:2
reshape_1/Reshape/shape?
reshape_1/ReshapeReshapeExtractImagePatches:patches:0 reshape_1/Reshape/shape:output:0*
T0*9
_output_shapes'
%:#???????????	2
reshape_1/Reshape?
MulMulreshape/Reshape:output:0reshape_1/Reshape:output:0*
T0*9
_output_shapes'
%:#???????????	2
Mulp
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indicesz
SumSumMul:z:0Sum/reduction_indices:output:0*
T0*5
_output_shapes#
!:???????????2
Sum^
reshape_2/ShapeShapeSum:output:0*
T0*
_output_shapes
:2
reshape_2/Shape?
reshape_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_2/strided_slice/stack?
reshape_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_2/strided_slice/stack_1?
reshape_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_2/strided_slice/stack_2?
reshape_2/strided_sliceStridedSlicereshape_2/Shape:output:0&reshape_2/strided_slice/stack:output:0(reshape_2/strided_slice/stack_1:output:0(reshape_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_2/strided_slicey
reshape_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :?2
reshape_2/Reshape/shape/1y
reshape_2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :?2
reshape_2/Reshape/shape/2x
reshape_2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_2/Reshape/shape/3?
reshape_2/Reshape/shapePack reshape_2/strided_slice:output:0"reshape_2/Reshape/shape/1:output:0"reshape_2/Reshape/shape/2:output:0"reshape_2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_2/Reshape/shape?
reshape_2/ReshapeReshapeSum:output:0 reshape_2/Reshape/shape:output:0*
T0*1
_output_shapes
:???????????2
reshape_2/Reshape?

Identity_1Identityreshape_2/Reshape:output:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity_1?
NoOpNoOp.^sequential/batch_normalization/AssignNewValue0^sequential/batch_normalization/AssignNewValue_1?^sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpA^sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1.^sequential/batch_normalization/ReadVariableOp0^sequential/batch_normalization/ReadVariableOp_1)^sequential/conv2d/BiasAdd/ReadVariableOp(^sequential/conv2d/Conv2D/ReadVariableOp+^sequential/conv2d_1/BiasAdd/ReadVariableOp*^sequential/conv2d_1/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:???????????: : : : : : : : 2^
-sequential/batch_normalization/AssignNewValue-sequential/batch_normalization/AssignNewValue2b
/sequential/batch_normalization/AssignNewValue_1/sequential/batch_normalization/AssignNewValue_12?
>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp2?
@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_12^
-sequential/batch_normalization/ReadVariableOp-sequential/batch_normalization/ReadVariableOp2b
/sequential/batch_normalization/ReadVariableOp_1/sequential/batch_normalization/ReadVariableOp_12T
(sequential/conv2d/BiasAdd/ReadVariableOp(sequential/conv2d/BiasAdd/ReadVariableOp2R
'sequential/conv2d/Conv2D/ReadVariableOp'sequential/conv2d/Conv2D/ReadVariableOp2X
*sequential/conv2d_1/BiasAdd/ReadVariableOp*sequential/conv2d_1/BiasAdd/ReadVariableOp2V
)sequential/conv2d_1/Conv2D/ReadVariableOp)sequential/conv2d_1/Conv2D/ReadVariableOp:T P
1
_output_shapes
:???????????

_user_specified_namex
?`
?
A__inference_inv_1_layer_call_and_return_conditional_losses_306070
xJ
0sequential_conv2d_conv2d_readvariableop_resource:?
1sequential_conv2d_biasadd_readvariableop_resource:D
6sequential_batch_normalization_readvariableop_resource:F
8sequential_batch_normalization_readvariableop_1_resource:U
Gsequential_batch_normalization_fusedbatchnormv3_readvariableop_resource:W
Isequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource:L
2sequential_conv2d_1_conv2d_readvariableop_resource:	A
3sequential_conv2d_1_biasadd_readvariableop_resource:	

identity_1??>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp?@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?-sequential/batch_normalization/ReadVariableOp?/sequential/batch_normalization/ReadVariableOp_1?(sequential/conv2d/BiasAdd/ReadVariableOp?'sequential/conv2d/Conv2D/ReadVariableOp?*sequential/conv2d_1/BiasAdd/ReadVariableOp?)sequential/conv2d_1/Conv2D/ReadVariableOp_
IdentityIdentityx*
T0*1
_output_shapes
:???????????2

Identity?
'sequential/conv2d/Conv2D/ReadVariableOpReadVariableOp0sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02)
'sequential/conv2d/Conv2D/ReadVariableOp?
sequential/conv2d/Conv2DConv2DIdentity:output:0/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
sequential/conv2d/Conv2D?
(sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOp1sequential_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(sequential/conv2d/BiasAdd/ReadVariableOp?
sequential/conv2d/BiasAddBiasAdd!sequential/conv2d/Conv2D:output:00sequential/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
sequential/conv2d/BiasAdd?
-sequential/batch_normalization/ReadVariableOpReadVariableOp6sequential_batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential/batch_normalization/ReadVariableOp?
/sequential/batch_normalization/ReadVariableOp_1ReadVariableOp8sequential_batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype021
/sequential/batch_normalization/ReadVariableOp_1?
>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpGsequential_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02@
>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp?
@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIsequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02B
@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?
/sequential/batch_normalization/FusedBatchNormV3FusedBatchNormV3"sequential/conv2d/BiasAdd:output:05sequential/batch_normalization/ReadVariableOp:value:07sequential/batch_normalization/ReadVariableOp_1:value:0Fsequential/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Hsequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( 21
/sequential/batch_normalization/FusedBatchNormV3?
sequential/re_lu/ReluRelu3sequential/batch_normalization/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????2
sequential/re_lu/Relu?
)sequential/conv2d_1/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:	*
dtype02+
)sequential/conv2d_1/Conv2D/ReadVariableOp?
sequential/conv2d_1/Conv2DConv2D#sequential/re_lu/Relu:activations:01sequential/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????	*
paddingVALID*
strides
2
sequential/conv2d_1/Conv2D?
*sequential/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02,
*sequential/conv2d_1/BiasAdd/ReadVariableOp?
sequential/conv2d_1/BiasAddBiasAdd#sequential/conv2d_1/Conv2D:output:02sequential/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????	2
sequential/conv2d_1/BiasAddr
reshape/ShapeShape$sequential/conv2d_1/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape/Shape?
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack?
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1?
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2?
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_sliceu
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :?2
reshape/Reshape/shape/1u
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :?2
reshape/Reshape/shape/2t
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :	2
reshape/Reshape/shape/3t
reshape/Reshape/shape/4Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/4t
reshape/Reshape/shape/5Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/5?
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0 reshape/Reshape/shape/4:output:0 reshape/Reshape/shape/5:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape?
reshape/ReshapeReshape$sequential/conv2d_1/BiasAdd:output:0reshape/Reshape/shape:output:0*
T0*9
_output_shapes'
%:#???????????	2
reshape/Reshape?
ExtractImagePatchesExtractImagePatchesx*
T0*1
_output_shapes
:???????????*
ksizes
*
paddingSAME*
rates
*
strides
2
ExtractImagePatcheso
reshape_1/ShapeShapeExtractImagePatches:patches:0*
T0*
_output_shapes
:2
reshape_1/Shape?
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_1/strided_slice/stack?
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_1?
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_2?
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_1/strided_slicey
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :?2
reshape_1/Reshape/shape/1y
reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :?2
reshape_1/Reshape/shape/2x
reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :	2
reshape_1/Reshape/shape/3x
reshape_1/Reshape/shape/4Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/4x
reshape_1/Reshape/shape/5Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/5?
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0"reshape_1/Reshape/shape/2:output:0"reshape_1/Reshape/shape/3:output:0"reshape_1/Reshape/shape/4:output:0"reshape_1/Reshape/shape/5:output:0*
N*
T0*
_output_shapes
:2
reshape_1/Reshape/shape?
reshape_1/ReshapeReshapeExtractImagePatches:patches:0 reshape_1/Reshape/shape:output:0*
T0*9
_output_shapes'
%:#???????????	2
reshape_1/Reshape?
MulMulreshape/Reshape:output:0reshape_1/Reshape:output:0*
T0*9
_output_shapes'
%:#???????????	2
Mulp
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indicesz
SumSumMul:z:0Sum/reduction_indices:output:0*
T0*5
_output_shapes#
!:???????????2
Sum^
reshape_2/ShapeShapeSum:output:0*
T0*
_output_shapes
:2
reshape_2/Shape?
reshape_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_2/strided_slice/stack?
reshape_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_2/strided_slice/stack_1?
reshape_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_2/strided_slice/stack_2?
reshape_2/strided_sliceStridedSlicereshape_2/Shape:output:0&reshape_2/strided_slice/stack:output:0(reshape_2/strided_slice/stack_1:output:0(reshape_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_2/strided_slicey
reshape_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :?2
reshape_2/Reshape/shape/1y
reshape_2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :?2
reshape_2/Reshape/shape/2x
reshape_2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_2/Reshape/shape/3?
reshape_2/Reshape/shapePack reshape_2/strided_slice:output:0"reshape_2/Reshape/shape/1:output:0"reshape_2/Reshape/shape/2:output:0"reshape_2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_2/Reshape/shape?
reshape_2/ReshapeReshapeSum:output:0 reshape_2/Reshape/shape:output:0*
T0*1
_output_shapes
:???????????2
reshape_2/Reshape?

Identity_1Identityreshape_2/Reshape:output:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity_1?
NoOpNoOp?^sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpA^sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1.^sequential/batch_normalization/ReadVariableOp0^sequential/batch_normalization/ReadVariableOp_1)^sequential/conv2d/BiasAdd/ReadVariableOp(^sequential/conv2d/Conv2D/ReadVariableOp+^sequential/conv2d_1/BiasAdd/ReadVariableOp*^sequential/conv2d_1/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:???????????: : : : : : : : 2?
>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp2?
@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_12^
-sequential/batch_normalization/ReadVariableOp-sequential/batch_normalization/ReadVariableOp2b
/sequential/batch_normalization/ReadVariableOp_1/sequential/batch_normalization/ReadVariableOp_12T
(sequential/conv2d/BiasAdd/ReadVariableOp(sequential/conv2d/BiasAdd/ReadVariableOp2R
'sequential/conv2d/Conv2D/ReadVariableOp'sequential/conv2d/Conv2D/ReadVariableOp2X
*sequential/conv2d_1/BiasAdd/ReadVariableOp*sequential/conv2d_1/BiasAdd/ReadVariableOp2V
)sequential/conv2d_1/Conv2D/ReadVariableOp)sequential/conv2d_1/Conv2D/ReadVariableOp:T P
1
_output_shapes
:???????????

_user_specified_namex
?

?
+__inference_sequential_layer_call_fn_302066
conv2d_input!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5:	
	unknown_6:	
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????	**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_3020472
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????	2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:???????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
1
_output_shapes
:???????????
&
_user_specified_nameconv2d_input
?
_
C__inference_flatten_layer_call_and_return_conditional_losses_304045

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????`  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????$2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????$2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
&__inference_inv_3_layer_call_fn_306811
x!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5:	
	unknown_6:	
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_inv_3_layer_call_and_return_conditional_losses_3042832
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
/
_output_shapes
:?????????

_user_specified_namex
?
e
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_302280

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
'__inference_conv2d_layer_call_fn_307327

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_3019902
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
]
A__inference_re_lu_layer_call_and_return_conditional_losses_303388

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_layer_call_and_return_conditional_losses_307571

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????88:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????882

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????88: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????88
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_layer_call_and_return_conditional_losses_301913

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_layer_call_and_return_conditional_losses_307915

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
4__inference_batch_normalization_layer_call_fn_307584

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_3023372
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
g
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_303942

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????88:W S
/
_output_shapes
:?????????88
 
_user_specified_nameinputs
?

?
+__inference_sequential_layer_call_fn_307308

inputs!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5:	
	unknown_6:	
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????	*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_3035432
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????	2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
'__inference_inv_22_layer_call_fn_306576
x!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????88*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_inv_22_layer_call_and_return_conditional_losses_3045332
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????882

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????pp: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
/
_output_shapes
:?????????pp

_user_specified_namex
?
]
A__inference_re_lu_layer_call_and_return_conditional_losses_307972

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
_
C__inference_re_lu_3_layer_call_and_return_conditional_losses_306816

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
F__inference_sequential_layer_call_and_return_conditional_losses_302047

inputs'
conv2d_301991:
conv2d_301993:(
batch_normalization_302014:(
batch_normalization_302016:(
batch_normalization_302018:(
batch_normalization_302020:)
conv2d_1_302041:	
conv2d_1_302043:	
identity??+batch_normalization/StatefulPartitionedCall?conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_301991conv2d_301993*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_3019902 
conv2d/StatefulPartitionedCall?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_302014batch_normalization_302016batch_normalization_302018batch_normalization_302020*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_3020132-
+batch_normalization/StatefulPartitionedCall?
re_lu/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_re_lu_layer_call_and_return_conditional_losses_3020282
re_lu/PartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallre_lu/PartitionedCall:output:0conv2d_1_302041conv2d_1_302043*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_3020402"
 conv2d_1/StatefulPartitionedCall?
IdentityIdentity)conv2d_1/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????	2

Identity?
NoOpNoOp,^batch_normalization/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:???????????: : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?/
?
F__inference_sequential_layer_call_and_return_conditional_losses_307048

inputs?
%conv2d_conv2d_readvariableop_resource:4
&conv2d_biasadd_readvariableop_resource:9
+batch_normalization_readvariableop_resource:;
-batch_normalization_readvariableop_1_resource:J
<batch_normalization_fusedbatchnormv3_readvariableop_resource:L
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource:A
'conv2d_1_conv2d_readvariableop_resource:6
(conv2d_1_biasadd_readvariableop_resource:
identity??"batch_normalization/AssignNewValue?$batch_normalization/AssignNewValue_1?3batch_normalization/FusedBatchNormV3/ReadVariableOp?5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?"batch_normalization/ReadVariableOp?$batch_normalization/ReadVariableOp_1?conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????88*
paddingVALID*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????882
conv2d/BiasAdd?
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype02$
"batch_normalization/ReadVariableOp?
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype02&
$batch_normalization/ReadVariableOp_1?
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOp?
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d/BiasAdd:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????88:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2&
$batch_normalization/FusedBatchNormV3?
"batch_normalization/AssignNewValueAssignVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource1batch_normalization/FusedBatchNormV3:batch_mean:04^batch_normalization/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02$
"batch_normalization/AssignNewValue?
$batch_normalization/AssignNewValue_1AssignVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource5batch_normalization/FusedBatchNormV3:batch_variance:06^batch_normalization/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02&
$batch_normalization/AssignNewValue_1?

re_lu/ReluRelu(batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????882

re_lu/Relu?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2Dre_lu/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????88*
paddingVALID*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????882
conv2d_1/BiasAdd|
IdentityIdentityconv2d_1/BiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????882

Identity?
NoOpNoOp#^batch_normalization/AssignNewValue%^batch_normalization/AssignNewValue_14^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_1^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????88: : : : : : : : 2H
"batch_normalization/AssignNewValue"batch_normalization/AssignNewValue2L
$batch_normalization/AssignNewValue_1$batch_normalization/AssignNewValue_12j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????88
 
_user_specified_nameinputs
??
?-
!__inference__wrapped_model_301847
input_1^
Dinrfnet_model_inv_1_sequential_conv2d_conv2d_readvariableop_resource:S
Einrfnet_model_inv_1_sequential_conv2d_biasadd_readvariableop_resource:X
Jinrfnet_model_inv_1_sequential_batch_normalization_readvariableop_resource:Z
Linrfnet_model_inv_1_sequential_batch_normalization_readvariableop_1_resource:i
[inrfnet_model_inv_1_sequential_batch_normalization_fusedbatchnormv3_readvariableop_resource:k
]inrfnet_model_inv_1_sequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource:`
Finrfnet_model_inv_1_sequential_conv2d_1_conv2d_readvariableop_resource:	U
Ginrfnet_model_inv_1_sequential_conv2d_1_biasadd_readvariableop_resource:	_
Einrfnet_model_inv_22_sequential_conv2d_conv2d_readvariableop_resource:T
Finrfnet_model_inv_22_sequential_conv2d_biasadd_readvariableop_resource:Y
Kinrfnet_model_inv_22_sequential_batch_normalization_readvariableop_resource:[
Minrfnet_model_inv_22_sequential_batch_normalization_readvariableop_1_resource:j
\inrfnet_model_inv_22_sequential_batch_normalization_fusedbatchnormv3_readvariableop_resource:l
^inrfnet_model_inv_22_sequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource:a
Ginrfnet_model_inv_22_sequential_conv2d_1_conv2d_readvariableop_resource:V
Hinrfnet_model_inv_22_sequential_conv2d_1_biasadd_readvariableop_resource:_
Einrfnet_model_inv_21_sequential_conv2d_conv2d_readvariableop_resource:T
Finrfnet_model_inv_21_sequential_conv2d_biasadd_readvariableop_resource:Y
Kinrfnet_model_inv_21_sequential_batch_normalization_readvariableop_resource:[
Minrfnet_model_inv_21_sequential_batch_normalization_readvariableop_1_resource:j
\inrfnet_model_inv_21_sequential_batch_normalization_fusedbatchnormv3_readvariableop_resource:l
^inrfnet_model_inv_21_sequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource:a
Ginrfnet_model_inv_21_sequential_conv2d_1_conv2d_readvariableop_resource:V
Hinrfnet_model_inv_21_sequential_conv2d_1_biasadd_readvariableop_resource:^
Dinrfnet_model_inv_3_sequential_conv2d_conv2d_readvariableop_resource:S
Einrfnet_model_inv_3_sequential_conv2d_biasadd_readvariableop_resource:X
Jinrfnet_model_inv_3_sequential_batch_normalization_readvariableop_resource:Z
Linrfnet_model_inv_3_sequential_batch_normalization_readvariableop_1_resource:i
[inrfnet_model_inv_3_sequential_batch_normalization_fusedbatchnormv3_readvariableop_resource:k
]inrfnet_model_inv_3_sequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource:`
Finrfnet_model_inv_3_sequential_conv2d_1_conv2d_readvariableop_resource:	U
Ginrfnet_model_inv_3_sequential_conv2d_1_biasadd_readvariableop_resource:	E
2inrfnet_model_dense_matmul_readvariableop_resource:	?$@A
3inrfnet_model_dense_biasadd_readvariableop_resource:@F
4inrfnet_model_dense_1_matmul_readvariableop_resource:@C
5inrfnet_model_dense_1_biasadd_readvariableop_resource:
identity??*InRFNet_Model/dense/BiasAdd/ReadVariableOp?)InRFNet_Model/dense/MatMul/ReadVariableOp?,InRFNet_Model/dense_1/BiasAdd/ReadVariableOp?+InRFNet_Model/dense_1/MatMul/ReadVariableOp?RInRFNet_Model/inv_1/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp?TInRFNet_Model/inv_1/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?AInRFNet_Model/inv_1/sequential/batch_normalization/ReadVariableOp?CInRFNet_Model/inv_1/sequential/batch_normalization/ReadVariableOp_1?<InRFNet_Model/inv_1/sequential/conv2d/BiasAdd/ReadVariableOp?;InRFNet_Model/inv_1/sequential/conv2d/Conv2D/ReadVariableOp?>InRFNet_Model/inv_1/sequential/conv2d_1/BiasAdd/ReadVariableOp?=InRFNet_Model/inv_1/sequential/conv2d_1/Conv2D/ReadVariableOp?SInRFNet_Model/inv_21/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp?UInRFNet_Model/inv_21/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?BInRFNet_Model/inv_21/sequential/batch_normalization/ReadVariableOp?DInRFNet_Model/inv_21/sequential/batch_normalization/ReadVariableOp_1?=InRFNet_Model/inv_21/sequential/conv2d/BiasAdd/ReadVariableOp?<InRFNet_Model/inv_21/sequential/conv2d/Conv2D/ReadVariableOp??InRFNet_Model/inv_21/sequential/conv2d_1/BiasAdd/ReadVariableOp?>InRFNet_Model/inv_21/sequential/conv2d_1/Conv2D/ReadVariableOp?SInRFNet_Model/inv_22/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp?UInRFNet_Model/inv_22/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?BInRFNet_Model/inv_22/sequential/batch_normalization/ReadVariableOp?DInRFNet_Model/inv_22/sequential/batch_normalization/ReadVariableOp_1?=InRFNet_Model/inv_22/sequential/conv2d/BiasAdd/ReadVariableOp?<InRFNet_Model/inv_22/sequential/conv2d/Conv2D/ReadVariableOp??InRFNet_Model/inv_22/sequential/conv2d_1/BiasAdd/ReadVariableOp?>InRFNet_Model/inv_22/sequential/conv2d_1/Conv2D/ReadVariableOp?RInRFNet_Model/inv_3/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp?TInRFNet_Model/inv_3/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?AInRFNet_Model/inv_3/sequential/batch_normalization/ReadVariableOp?CInRFNet_Model/inv_3/sequential/batch_normalization/ReadVariableOp_1?<InRFNet_Model/inv_3/sequential/conv2d/BiasAdd/ReadVariableOp?;InRFNet_Model/inv_3/sequential/conv2d/Conv2D/ReadVariableOp?>InRFNet_Model/inv_3/sequential/conv2d_1/BiasAdd/ReadVariableOp?=InRFNet_Model/inv_3/sequential/conv2d_1/Conv2D/ReadVariableOp?
InRFNet_Model/inv_1/IdentityIdentityinput_1*
T0*1
_output_shapes
:???????????2
InRFNet_Model/inv_1/Identity?
;InRFNet_Model/inv_1/sequential/conv2d/Conv2D/ReadVariableOpReadVariableOpDinrfnet_model_inv_1_sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02=
;InRFNet_Model/inv_1/sequential/conv2d/Conv2D/ReadVariableOp?
,InRFNet_Model/inv_1/sequential/conv2d/Conv2DConv2D%InRFNet_Model/inv_1/Identity:output:0CInRFNet_Model/inv_1/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2.
,InRFNet_Model/inv_1/sequential/conv2d/Conv2D?
<InRFNet_Model/inv_1/sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOpEinrfnet_model_inv_1_sequential_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02>
<InRFNet_Model/inv_1/sequential/conv2d/BiasAdd/ReadVariableOp?
-InRFNet_Model/inv_1/sequential/conv2d/BiasAddBiasAdd5InRFNet_Model/inv_1/sequential/conv2d/Conv2D:output:0DInRFNet_Model/inv_1/sequential/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2/
-InRFNet_Model/inv_1/sequential/conv2d/BiasAdd?
AInRFNet_Model/inv_1/sequential/batch_normalization/ReadVariableOpReadVariableOpJinrfnet_model_inv_1_sequential_batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype02C
AInRFNet_Model/inv_1/sequential/batch_normalization/ReadVariableOp?
CInRFNet_Model/inv_1/sequential/batch_normalization/ReadVariableOp_1ReadVariableOpLinrfnet_model_inv_1_sequential_batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype02E
CInRFNet_Model/inv_1/sequential/batch_normalization/ReadVariableOp_1?
RInRFNet_Model/inv_1/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp[inrfnet_model_inv_1_sequential_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02T
RInRFNet_Model/inv_1/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp?
TInRFNet_Model/inv_1/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]inrfnet_model_inv_1_sequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02V
TInRFNet_Model/inv_1/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?
CInRFNet_Model/inv_1/sequential/batch_normalization/FusedBatchNormV3FusedBatchNormV36InRFNet_Model/inv_1/sequential/conv2d/BiasAdd:output:0IInRFNet_Model/inv_1/sequential/batch_normalization/ReadVariableOp:value:0KInRFNet_Model/inv_1/sequential/batch_normalization/ReadVariableOp_1:value:0ZInRFNet_Model/inv_1/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0\InRFNet_Model/inv_1/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( 2E
CInRFNet_Model/inv_1/sequential/batch_normalization/FusedBatchNormV3?
)InRFNet_Model/inv_1/sequential/re_lu/ReluReluGInRFNet_Model/inv_1/sequential/batch_normalization/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????2+
)InRFNet_Model/inv_1/sequential/re_lu/Relu?
=InRFNet_Model/inv_1/sequential/conv2d_1/Conv2D/ReadVariableOpReadVariableOpFinrfnet_model_inv_1_sequential_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:	*
dtype02?
=InRFNet_Model/inv_1/sequential/conv2d_1/Conv2D/ReadVariableOp?
.InRFNet_Model/inv_1/sequential/conv2d_1/Conv2DConv2D7InRFNet_Model/inv_1/sequential/re_lu/Relu:activations:0EInRFNet_Model/inv_1/sequential/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????	*
paddingVALID*
strides
20
.InRFNet_Model/inv_1/sequential/conv2d_1/Conv2D?
>InRFNet_Model/inv_1/sequential/conv2d_1/BiasAdd/ReadVariableOpReadVariableOpGinrfnet_model_inv_1_sequential_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02@
>InRFNet_Model/inv_1/sequential/conv2d_1/BiasAdd/ReadVariableOp?
/InRFNet_Model/inv_1/sequential/conv2d_1/BiasAddBiasAdd7InRFNet_Model/inv_1/sequential/conv2d_1/Conv2D:output:0FInRFNet_Model/inv_1/sequential/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????	21
/InRFNet_Model/inv_1/sequential/conv2d_1/BiasAdd?
!InRFNet_Model/inv_1/reshape/ShapeShape8InRFNet_Model/inv_1/sequential/conv2d_1/BiasAdd:output:0*
T0*
_output_shapes
:2#
!InRFNet_Model/inv_1/reshape/Shape?
/InRFNet_Model/inv_1/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/InRFNet_Model/inv_1/reshape/strided_slice/stack?
1InRFNet_Model/inv_1/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1InRFNet_Model/inv_1/reshape/strided_slice/stack_1?
1InRFNet_Model/inv_1/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1InRFNet_Model/inv_1/reshape/strided_slice/stack_2?
)InRFNet_Model/inv_1/reshape/strided_sliceStridedSlice*InRFNet_Model/inv_1/reshape/Shape:output:08InRFNet_Model/inv_1/reshape/strided_slice/stack:output:0:InRFNet_Model/inv_1/reshape/strided_slice/stack_1:output:0:InRFNet_Model/inv_1/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)InRFNet_Model/inv_1/reshape/strided_slice?
+InRFNet_Model/inv_1/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :?2-
+InRFNet_Model/inv_1/reshape/Reshape/shape/1?
+InRFNet_Model/inv_1/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :?2-
+InRFNet_Model/inv_1/reshape/Reshape/shape/2?
+InRFNet_Model/inv_1/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :	2-
+InRFNet_Model/inv_1/reshape/Reshape/shape/3?
+InRFNet_Model/inv_1/reshape/Reshape/shape/4Const*
_output_shapes
: *
dtype0*
value	B :2-
+InRFNet_Model/inv_1/reshape/Reshape/shape/4?
+InRFNet_Model/inv_1/reshape/Reshape/shape/5Const*
_output_shapes
: *
dtype0*
value	B :2-
+InRFNet_Model/inv_1/reshape/Reshape/shape/5?
)InRFNet_Model/inv_1/reshape/Reshape/shapePack2InRFNet_Model/inv_1/reshape/strided_slice:output:04InRFNet_Model/inv_1/reshape/Reshape/shape/1:output:04InRFNet_Model/inv_1/reshape/Reshape/shape/2:output:04InRFNet_Model/inv_1/reshape/Reshape/shape/3:output:04InRFNet_Model/inv_1/reshape/Reshape/shape/4:output:04InRFNet_Model/inv_1/reshape/Reshape/shape/5:output:0*
N*
T0*
_output_shapes
:2+
)InRFNet_Model/inv_1/reshape/Reshape/shape?
#InRFNet_Model/inv_1/reshape/ReshapeReshape8InRFNet_Model/inv_1/sequential/conv2d_1/BiasAdd:output:02InRFNet_Model/inv_1/reshape/Reshape/shape:output:0*
T0*9
_output_shapes'
%:#???????????	2%
#InRFNet_Model/inv_1/reshape/Reshape?
'InRFNet_Model/inv_1/ExtractImagePatchesExtractImagePatchesinput_1*
T0*1
_output_shapes
:???????????*
ksizes
*
paddingSAME*
rates
*
strides
2)
'InRFNet_Model/inv_1/ExtractImagePatches?
#InRFNet_Model/inv_1/reshape_1/ShapeShape1InRFNet_Model/inv_1/ExtractImagePatches:patches:0*
T0*
_output_shapes
:2%
#InRFNet_Model/inv_1/reshape_1/Shape?
1InRFNet_Model/inv_1/reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1InRFNet_Model/inv_1/reshape_1/strided_slice/stack?
3InRFNet_Model/inv_1/reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3InRFNet_Model/inv_1/reshape_1/strided_slice/stack_1?
3InRFNet_Model/inv_1/reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3InRFNet_Model/inv_1/reshape_1/strided_slice/stack_2?
+InRFNet_Model/inv_1/reshape_1/strided_sliceStridedSlice,InRFNet_Model/inv_1/reshape_1/Shape:output:0:InRFNet_Model/inv_1/reshape_1/strided_slice/stack:output:0<InRFNet_Model/inv_1/reshape_1/strided_slice/stack_1:output:0<InRFNet_Model/inv_1/reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+InRFNet_Model/inv_1/reshape_1/strided_slice?
-InRFNet_Model/inv_1/reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :?2/
-InRFNet_Model/inv_1/reshape_1/Reshape/shape/1?
-InRFNet_Model/inv_1/reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :?2/
-InRFNet_Model/inv_1/reshape_1/Reshape/shape/2?
-InRFNet_Model/inv_1/reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :	2/
-InRFNet_Model/inv_1/reshape_1/Reshape/shape/3?
-InRFNet_Model/inv_1/reshape_1/Reshape/shape/4Const*
_output_shapes
: *
dtype0*
value	B :2/
-InRFNet_Model/inv_1/reshape_1/Reshape/shape/4?
-InRFNet_Model/inv_1/reshape_1/Reshape/shape/5Const*
_output_shapes
: *
dtype0*
value	B :2/
-InRFNet_Model/inv_1/reshape_1/Reshape/shape/5?
+InRFNet_Model/inv_1/reshape_1/Reshape/shapePack4InRFNet_Model/inv_1/reshape_1/strided_slice:output:06InRFNet_Model/inv_1/reshape_1/Reshape/shape/1:output:06InRFNet_Model/inv_1/reshape_1/Reshape/shape/2:output:06InRFNet_Model/inv_1/reshape_1/Reshape/shape/3:output:06InRFNet_Model/inv_1/reshape_1/Reshape/shape/4:output:06InRFNet_Model/inv_1/reshape_1/Reshape/shape/5:output:0*
N*
T0*
_output_shapes
:2-
+InRFNet_Model/inv_1/reshape_1/Reshape/shape?
%InRFNet_Model/inv_1/reshape_1/ReshapeReshape1InRFNet_Model/inv_1/ExtractImagePatches:patches:04InRFNet_Model/inv_1/reshape_1/Reshape/shape:output:0*
T0*9
_output_shapes'
%:#???????????	2'
%InRFNet_Model/inv_1/reshape_1/Reshape?
InRFNet_Model/inv_1/MulMul,InRFNet_Model/inv_1/reshape/Reshape:output:0.InRFNet_Model/inv_1/reshape_1/Reshape:output:0*
T0*9
_output_shapes'
%:#???????????	2
InRFNet_Model/inv_1/Mul?
)InRFNet_Model/inv_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2+
)InRFNet_Model/inv_1/Sum/reduction_indices?
InRFNet_Model/inv_1/SumSumInRFNet_Model/inv_1/Mul:z:02InRFNet_Model/inv_1/Sum/reduction_indices:output:0*
T0*5
_output_shapes#
!:???????????2
InRFNet_Model/inv_1/Sum?
#InRFNet_Model/inv_1/reshape_2/ShapeShape InRFNet_Model/inv_1/Sum:output:0*
T0*
_output_shapes
:2%
#InRFNet_Model/inv_1/reshape_2/Shape?
1InRFNet_Model/inv_1/reshape_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1InRFNet_Model/inv_1/reshape_2/strided_slice/stack?
3InRFNet_Model/inv_1/reshape_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3InRFNet_Model/inv_1/reshape_2/strided_slice/stack_1?
3InRFNet_Model/inv_1/reshape_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3InRFNet_Model/inv_1/reshape_2/strided_slice/stack_2?
+InRFNet_Model/inv_1/reshape_2/strided_sliceStridedSlice,InRFNet_Model/inv_1/reshape_2/Shape:output:0:InRFNet_Model/inv_1/reshape_2/strided_slice/stack:output:0<InRFNet_Model/inv_1/reshape_2/strided_slice/stack_1:output:0<InRFNet_Model/inv_1/reshape_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+InRFNet_Model/inv_1/reshape_2/strided_slice?
-InRFNet_Model/inv_1/reshape_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :?2/
-InRFNet_Model/inv_1/reshape_2/Reshape/shape/1?
-InRFNet_Model/inv_1/reshape_2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :?2/
-InRFNet_Model/inv_1/reshape_2/Reshape/shape/2?
-InRFNet_Model/inv_1/reshape_2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2/
-InRFNet_Model/inv_1/reshape_2/Reshape/shape/3?
+InRFNet_Model/inv_1/reshape_2/Reshape/shapePack4InRFNet_Model/inv_1/reshape_2/strided_slice:output:06InRFNet_Model/inv_1/reshape_2/Reshape/shape/1:output:06InRFNet_Model/inv_1/reshape_2/Reshape/shape/2:output:06InRFNet_Model/inv_1/reshape_2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2-
+InRFNet_Model/inv_1/reshape_2/Reshape/shape?
%InRFNet_Model/inv_1/reshape_2/ReshapeReshape InRFNet_Model/inv_1/Sum:output:04InRFNet_Model/inv_1/reshape_2/Reshape/shape:output:0*
T0*1
_output_shapes
:???????????2'
%InRFNet_Model/inv_1/reshape_2/Reshape?
InRFNet_Model/re_lu/ReluRelu.InRFNet_Model/inv_1/reshape_2/Reshape:output:0*
T0*1
_output_shapes
:???????????2
InRFNet_Model/re_lu/Relu?
#InRFNet_Model/max_pooling2d/MaxPoolMaxPool&InRFNet_Model/re_lu/Relu:activations:0*/
_output_shapes
:?????????pp*
ksize
*
paddingVALID*
strides
2%
#InRFNet_Model/max_pooling2d/MaxPool?
.InRFNet_Model/inv_22/average_pooling2d/AvgPoolAvgPool,InRFNet_Model/max_pooling2d/MaxPool:output:0*
T0*/
_output_shapes
:?????????88*
ksize
*
paddingSAME*
strides
20
.InRFNet_Model/inv_22/average_pooling2d/AvgPool?
<InRFNet_Model/inv_22/sequential/conv2d/Conv2D/ReadVariableOpReadVariableOpEinrfnet_model_inv_22_sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02>
<InRFNet_Model/inv_22/sequential/conv2d/Conv2D/ReadVariableOp?
-InRFNet_Model/inv_22/sequential/conv2d/Conv2DConv2D7InRFNet_Model/inv_22/average_pooling2d/AvgPool:output:0DInRFNet_Model/inv_22/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????88*
paddingVALID*
strides
2/
-InRFNet_Model/inv_22/sequential/conv2d/Conv2D?
=InRFNet_Model/inv_22/sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOpFinrfnet_model_inv_22_sequential_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02?
=InRFNet_Model/inv_22/sequential/conv2d/BiasAdd/ReadVariableOp?
.InRFNet_Model/inv_22/sequential/conv2d/BiasAddBiasAdd6InRFNet_Model/inv_22/sequential/conv2d/Conv2D:output:0EInRFNet_Model/inv_22/sequential/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????8820
.InRFNet_Model/inv_22/sequential/conv2d/BiasAdd?
BInRFNet_Model/inv_22/sequential/batch_normalization/ReadVariableOpReadVariableOpKinrfnet_model_inv_22_sequential_batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype02D
BInRFNet_Model/inv_22/sequential/batch_normalization/ReadVariableOp?
DInRFNet_Model/inv_22/sequential/batch_normalization/ReadVariableOp_1ReadVariableOpMinrfnet_model_inv_22_sequential_batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype02F
DInRFNet_Model/inv_22/sequential/batch_normalization/ReadVariableOp_1?
SInRFNet_Model/inv_22/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp\inrfnet_model_inv_22_sequential_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02U
SInRFNet_Model/inv_22/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp?
UInRFNet_Model/inv_22/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp^inrfnet_model_inv_22_sequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02W
UInRFNet_Model/inv_22/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?
DInRFNet_Model/inv_22/sequential/batch_normalization/FusedBatchNormV3FusedBatchNormV37InRFNet_Model/inv_22/sequential/conv2d/BiasAdd:output:0JInRFNet_Model/inv_22/sequential/batch_normalization/ReadVariableOp:value:0LInRFNet_Model/inv_22/sequential/batch_normalization/ReadVariableOp_1:value:0[InRFNet_Model/inv_22/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0]InRFNet_Model/inv_22/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????88:::::*
epsilon%o?:*
is_training( 2F
DInRFNet_Model/inv_22/sequential/batch_normalization/FusedBatchNormV3?
*InRFNet_Model/inv_22/sequential/re_lu/ReluReluHInRFNet_Model/inv_22/sequential/batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????882,
*InRFNet_Model/inv_22/sequential/re_lu/Relu?
>InRFNet_Model/inv_22/sequential/conv2d_1/Conv2D/ReadVariableOpReadVariableOpGinrfnet_model_inv_22_sequential_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02@
>InRFNet_Model/inv_22/sequential/conv2d_1/Conv2D/ReadVariableOp?
/InRFNet_Model/inv_22/sequential/conv2d_1/Conv2DConv2D8InRFNet_Model/inv_22/sequential/re_lu/Relu:activations:0FInRFNet_Model/inv_22/sequential/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????88*
paddingVALID*
strides
21
/InRFNet_Model/inv_22/sequential/conv2d_1/Conv2D?
?InRFNet_Model/inv_22/sequential/conv2d_1/BiasAdd/ReadVariableOpReadVariableOpHinrfnet_model_inv_22_sequential_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02A
?InRFNet_Model/inv_22/sequential/conv2d_1/BiasAdd/ReadVariableOp?
0InRFNet_Model/inv_22/sequential/conv2d_1/BiasAddBiasAdd8InRFNet_Model/inv_22/sequential/conv2d_1/Conv2D:output:0GInRFNet_Model/inv_22/sequential/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????8822
0InRFNet_Model/inv_22/sequential/conv2d_1/BiasAdd?
"InRFNet_Model/inv_22/reshape/ShapeShape9InRFNet_Model/inv_22/sequential/conv2d_1/BiasAdd:output:0*
T0*
_output_shapes
:2$
"InRFNet_Model/inv_22/reshape/Shape?
0InRFNet_Model/inv_22/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0InRFNet_Model/inv_22/reshape/strided_slice/stack?
2InRFNet_Model/inv_22/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2InRFNet_Model/inv_22/reshape/strided_slice/stack_1?
2InRFNet_Model/inv_22/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2InRFNet_Model/inv_22/reshape/strided_slice/stack_2?
*InRFNet_Model/inv_22/reshape/strided_sliceStridedSlice+InRFNet_Model/inv_22/reshape/Shape:output:09InRFNet_Model/inv_22/reshape/strided_slice/stack:output:0;InRFNet_Model/inv_22/reshape/strided_slice/stack_1:output:0;InRFNet_Model/inv_22/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*InRFNet_Model/inv_22/reshape/strided_slice?
,InRFNet_Model/inv_22/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :82.
,InRFNet_Model/inv_22/reshape/Reshape/shape/1?
,InRFNet_Model/inv_22/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :82.
,InRFNet_Model/inv_22/reshape/Reshape/shape/2?
,InRFNet_Model/inv_22/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2.
,InRFNet_Model/inv_22/reshape/Reshape/shape/3?
,InRFNet_Model/inv_22/reshape/Reshape/shape/4Const*
_output_shapes
: *
dtype0*
value	B :2.
,InRFNet_Model/inv_22/reshape/Reshape/shape/4?
,InRFNet_Model/inv_22/reshape/Reshape/shape/5Const*
_output_shapes
: *
dtype0*
value	B :2.
,InRFNet_Model/inv_22/reshape/Reshape/shape/5?
*InRFNet_Model/inv_22/reshape/Reshape/shapePack3InRFNet_Model/inv_22/reshape/strided_slice:output:05InRFNet_Model/inv_22/reshape/Reshape/shape/1:output:05InRFNet_Model/inv_22/reshape/Reshape/shape/2:output:05InRFNet_Model/inv_22/reshape/Reshape/shape/3:output:05InRFNet_Model/inv_22/reshape/Reshape/shape/4:output:05InRFNet_Model/inv_22/reshape/Reshape/shape/5:output:0*
N*
T0*
_output_shapes
:2,
*InRFNet_Model/inv_22/reshape/Reshape/shape?
$InRFNet_Model/inv_22/reshape/ReshapeReshape9InRFNet_Model/inv_22/sequential/conv2d_1/BiasAdd:output:03InRFNet_Model/inv_22/reshape/Reshape/shape:output:0*
T0*7
_output_shapes%
#:!?????????882&
$InRFNet_Model/inv_22/reshape/Reshape?
(InRFNet_Model/inv_22/ExtractImagePatchesExtractImagePatches,InRFNet_Model/max_pooling2d/MaxPool:output:0*
T0*/
_output_shapes
:?????????88	*
ksizes
*
paddingSAME*
rates
*
strides
2*
(InRFNet_Model/inv_22/ExtractImagePatches?
$InRFNet_Model/inv_22/reshape_1/ShapeShape2InRFNet_Model/inv_22/ExtractImagePatches:patches:0*
T0*
_output_shapes
:2&
$InRFNet_Model/inv_22/reshape_1/Shape?
2InRFNet_Model/inv_22/reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 24
2InRFNet_Model/inv_22/reshape_1/strided_slice/stack?
4InRFNet_Model/inv_22/reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:26
4InRFNet_Model/inv_22/reshape_1/strided_slice/stack_1?
4InRFNet_Model/inv_22/reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4InRFNet_Model/inv_22/reshape_1/strided_slice/stack_2?
,InRFNet_Model/inv_22/reshape_1/strided_sliceStridedSlice-InRFNet_Model/inv_22/reshape_1/Shape:output:0;InRFNet_Model/inv_22/reshape_1/strided_slice/stack:output:0=InRFNet_Model/inv_22/reshape_1/strided_slice/stack_1:output:0=InRFNet_Model/inv_22/reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,InRFNet_Model/inv_22/reshape_1/strided_slice?
.InRFNet_Model/inv_22/reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :820
.InRFNet_Model/inv_22/reshape_1/Reshape/shape/1?
.InRFNet_Model/inv_22/reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :820
.InRFNet_Model/inv_22/reshape_1/Reshape/shape/2?
.InRFNet_Model/inv_22/reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :20
.InRFNet_Model/inv_22/reshape_1/Reshape/shape/3?
.InRFNet_Model/inv_22/reshape_1/Reshape/shape/4Const*
_output_shapes
: *
dtype0*
value	B :20
.InRFNet_Model/inv_22/reshape_1/Reshape/shape/4?
.InRFNet_Model/inv_22/reshape_1/Reshape/shape/5Const*
_output_shapes
: *
dtype0*
value	B :20
.InRFNet_Model/inv_22/reshape_1/Reshape/shape/5?
,InRFNet_Model/inv_22/reshape_1/Reshape/shapePack5InRFNet_Model/inv_22/reshape_1/strided_slice:output:07InRFNet_Model/inv_22/reshape_1/Reshape/shape/1:output:07InRFNet_Model/inv_22/reshape_1/Reshape/shape/2:output:07InRFNet_Model/inv_22/reshape_1/Reshape/shape/3:output:07InRFNet_Model/inv_22/reshape_1/Reshape/shape/4:output:07InRFNet_Model/inv_22/reshape_1/Reshape/shape/5:output:0*
N*
T0*
_output_shapes
:2.
,InRFNet_Model/inv_22/reshape_1/Reshape/shape?
&InRFNet_Model/inv_22/reshape_1/ReshapeReshape2InRFNet_Model/inv_22/ExtractImagePatches:patches:05InRFNet_Model/inv_22/reshape_1/Reshape/shape:output:0*
T0*7
_output_shapes%
#:!?????????882(
&InRFNet_Model/inv_22/reshape_1/Reshape?
InRFNet_Model/inv_22/MulMul-InRFNet_Model/inv_22/reshape/Reshape:output:0/InRFNet_Model/inv_22/reshape_1/Reshape:output:0*
T0*7
_output_shapes%
#:!?????????882
InRFNet_Model/inv_22/Mul?
*InRFNet_Model/inv_22/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*InRFNet_Model/inv_22/Sum/reduction_indices?
InRFNet_Model/inv_22/SumSumInRFNet_Model/inv_22/Mul:z:03InRFNet_Model/inv_22/Sum/reduction_indices:output:0*
T0*3
_output_shapes!
:?????????882
InRFNet_Model/inv_22/Sum?
$InRFNet_Model/inv_22/reshape_2/ShapeShape!InRFNet_Model/inv_22/Sum:output:0*
T0*
_output_shapes
:2&
$InRFNet_Model/inv_22/reshape_2/Shape?
2InRFNet_Model/inv_22/reshape_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 24
2InRFNet_Model/inv_22/reshape_2/strided_slice/stack?
4InRFNet_Model/inv_22/reshape_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:26
4InRFNet_Model/inv_22/reshape_2/strided_slice/stack_1?
4InRFNet_Model/inv_22/reshape_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4InRFNet_Model/inv_22/reshape_2/strided_slice/stack_2?
,InRFNet_Model/inv_22/reshape_2/strided_sliceStridedSlice-InRFNet_Model/inv_22/reshape_2/Shape:output:0;InRFNet_Model/inv_22/reshape_2/strided_slice/stack:output:0=InRFNet_Model/inv_22/reshape_2/strided_slice/stack_1:output:0=InRFNet_Model/inv_22/reshape_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,InRFNet_Model/inv_22/reshape_2/strided_slice?
.InRFNet_Model/inv_22/reshape_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :820
.InRFNet_Model/inv_22/reshape_2/Reshape/shape/1?
.InRFNet_Model/inv_22/reshape_2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :820
.InRFNet_Model/inv_22/reshape_2/Reshape/shape/2?
.InRFNet_Model/inv_22/reshape_2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :20
.InRFNet_Model/inv_22/reshape_2/Reshape/shape/3?
,InRFNet_Model/inv_22/reshape_2/Reshape/shapePack5InRFNet_Model/inv_22/reshape_2/strided_slice:output:07InRFNet_Model/inv_22/reshape_2/Reshape/shape/1:output:07InRFNet_Model/inv_22/reshape_2/Reshape/shape/2:output:07InRFNet_Model/inv_22/reshape_2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2.
,InRFNet_Model/inv_22/reshape_2/Reshape/shape?
&InRFNet_Model/inv_22/reshape_2/ReshapeReshape!InRFNet_Model/inv_22/Sum:output:05InRFNet_Model/inv_22/reshape_2/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????882(
&InRFNet_Model/inv_22/reshape_2/Reshape?
.InRFNet_Model/inv_21/average_pooling2d/AvgPoolAvgPool,InRFNet_Model/max_pooling2d/MaxPool:output:0*
T0*/
_output_shapes
:?????????88*
ksize
*
paddingSAME*
strides
20
.InRFNet_Model/inv_21/average_pooling2d/AvgPool?
<InRFNet_Model/inv_21/sequential/conv2d/Conv2D/ReadVariableOpReadVariableOpEinrfnet_model_inv_21_sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02>
<InRFNet_Model/inv_21/sequential/conv2d/Conv2D/ReadVariableOp?
-InRFNet_Model/inv_21/sequential/conv2d/Conv2DConv2D7InRFNet_Model/inv_21/average_pooling2d/AvgPool:output:0DInRFNet_Model/inv_21/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????88*
paddingVALID*
strides
2/
-InRFNet_Model/inv_21/sequential/conv2d/Conv2D?
=InRFNet_Model/inv_21/sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOpFinrfnet_model_inv_21_sequential_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02?
=InRFNet_Model/inv_21/sequential/conv2d/BiasAdd/ReadVariableOp?
.InRFNet_Model/inv_21/sequential/conv2d/BiasAddBiasAdd6InRFNet_Model/inv_21/sequential/conv2d/Conv2D:output:0EInRFNet_Model/inv_21/sequential/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????8820
.InRFNet_Model/inv_21/sequential/conv2d/BiasAdd?
BInRFNet_Model/inv_21/sequential/batch_normalization/ReadVariableOpReadVariableOpKinrfnet_model_inv_21_sequential_batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype02D
BInRFNet_Model/inv_21/sequential/batch_normalization/ReadVariableOp?
DInRFNet_Model/inv_21/sequential/batch_normalization/ReadVariableOp_1ReadVariableOpMinrfnet_model_inv_21_sequential_batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype02F
DInRFNet_Model/inv_21/sequential/batch_normalization/ReadVariableOp_1?
SInRFNet_Model/inv_21/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp\inrfnet_model_inv_21_sequential_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02U
SInRFNet_Model/inv_21/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp?
UInRFNet_Model/inv_21/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp^inrfnet_model_inv_21_sequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02W
UInRFNet_Model/inv_21/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?
DInRFNet_Model/inv_21/sequential/batch_normalization/FusedBatchNormV3FusedBatchNormV37InRFNet_Model/inv_21/sequential/conv2d/BiasAdd:output:0JInRFNet_Model/inv_21/sequential/batch_normalization/ReadVariableOp:value:0LInRFNet_Model/inv_21/sequential/batch_normalization/ReadVariableOp_1:value:0[InRFNet_Model/inv_21/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0]InRFNet_Model/inv_21/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????88:::::*
epsilon%o?:*
is_training( 2F
DInRFNet_Model/inv_21/sequential/batch_normalization/FusedBatchNormV3?
*InRFNet_Model/inv_21/sequential/re_lu/ReluReluHInRFNet_Model/inv_21/sequential/batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????882,
*InRFNet_Model/inv_21/sequential/re_lu/Relu?
>InRFNet_Model/inv_21/sequential/conv2d_1/Conv2D/ReadVariableOpReadVariableOpGinrfnet_model_inv_21_sequential_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02@
>InRFNet_Model/inv_21/sequential/conv2d_1/Conv2D/ReadVariableOp?
/InRFNet_Model/inv_21/sequential/conv2d_1/Conv2DConv2D8InRFNet_Model/inv_21/sequential/re_lu/Relu:activations:0FInRFNet_Model/inv_21/sequential/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????88*
paddingVALID*
strides
21
/InRFNet_Model/inv_21/sequential/conv2d_1/Conv2D?
?InRFNet_Model/inv_21/sequential/conv2d_1/BiasAdd/ReadVariableOpReadVariableOpHinrfnet_model_inv_21_sequential_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02A
?InRFNet_Model/inv_21/sequential/conv2d_1/BiasAdd/ReadVariableOp?
0InRFNet_Model/inv_21/sequential/conv2d_1/BiasAddBiasAdd8InRFNet_Model/inv_21/sequential/conv2d_1/Conv2D:output:0GInRFNet_Model/inv_21/sequential/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????8822
0InRFNet_Model/inv_21/sequential/conv2d_1/BiasAdd?
"InRFNet_Model/inv_21/reshape/ShapeShape9InRFNet_Model/inv_21/sequential/conv2d_1/BiasAdd:output:0*
T0*
_output_shapes
:2$
"InRFNet_Model/inv_21/reshape/Shape?
0InRFNet_Model/inv_21/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0InRFNet_Model/inv_21/reshape/strided_slice/stack?
2InRFNet_Model/inv_21/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2InRFNet_Model/inv_21/reshape/strided_slice/stack_1?
2InRFNet_Model/inv_21/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2InRFNet_Model/inv_21/reshape/strided_slice/stack_2?
*InRFNet_Model/inv_21/reshape/strided_sliceStridedSlice+InRFNet_Model/inv_21/reshape/Shape:output:09InRFNet_Model/inv_21/reshape/strided_slice/stack:output:0;InRFNet_Model/inv_21/reshape/strided_slice/stack_1:output:0;InRFNet_Model/inv_21/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*InRFNet_Model/inv_21/reshape/strided_slice?
,InRFNet_Model/inv_21/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :82.
,InRFNet_Model/inv_21/reshape/Reshape/shape/1?
,InRFNet_Model/inv_21/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :82.
,InRFNet_Model/inv_21/reshape/Reshape/shape/2?
,InRFNet_Model/inv_21/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2.
,InRFNet_Model/inv_21/reshape/Reshape/shape/3?
,InRFNet_Model/inv_21/reshape/Reshape/shape/4Const*
_output_shapes
: *
dtype0*
value	B :2.
,InRFNet_Model/inv_21/reshape/Reshape/shape/4?
,InRFNet_Model/inv_21/reshape/Reshape/shape/5Const*
_output_shapes
: *
dtype0*
value	B :2.
,InRFNet_Model/inv_21/reshape/Reshape/shape/5?
*InRFNet_Model/inv_21/reshape/Reshape/shapePack3InRFNet_Model/inv_21/reshape/strided_slice:output:05InRFNet_Model/inv_21/reshape/Reshape/shape/1:output:05InRFNet_Model/inv_21/reshape/Reshape/shape/2:output:05InRFNet_Model/inv_21/reshape/Reshape/shape/3:output:05InRFNet_Model/inv_21/reshape/Reshape/shape/4:output:05InRFNet_Model/inv_21/reshape/Reshape/shape/5:output:0*
N*
T0*
_output_shapes
:2,
*InRFNet_Model/inv_21/reshape/Reshape/shape?
$InRFNet_Model/inv_21/reshape/ReshapeReshape9InRFNet_Model/inv_21/sequential/conv2d_1/BiasAdd:output:03InRFNet_Model/inv_21/reshape/Reshape/shape:output:0*
T0*7
_output_shapes%
#:!?????????882&
$InRFNet_Model/inv_21/reshape/Reshape?
(InRFNet_Model/inv_21/ExtractImagePatchesExtractImagePatches,InRFNet_Model/max_pooling2d/MaxPool:output:0*
T0*/
_output_shapes
:?????????88	*
ksizes
*
paddingSAME*
rates
*
strides
2*
(InRFNet_Model/inv_21/ExtractImagePatches?
$InRFNet_Model/inv_21/reshape_1/ShapeShape2InRFNet_Model/inv_21/ExtractImagePatches:patches:0*
T0*
_output_shapes
:2&
$InRFNet_Model/inv_21/reshape_1/Shape?
2InRFNet_Model/inv_21/reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 24
2InRFNet_Model/inv_21/reshape_1/strided_slice/stack?
4InRFNet_Model/inv_21/reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:26
4InRFNet_Model/inv_21/reshape_1/strided_slice/stack_1?
4InRFNet_Model/inv_21/reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4InRFNet_Model/inv_21/reshape_1/strided_slice/stack_2?
,InRFNet_Model/inv_21/reshape_1/strided_sliceStridedSlice-InRFNet_Model/inv_21/reshape_1/Shape:output:0;InRFNet_Model/inv_21/reshape_1/strided_slice/stack:output:0=InRFNet_Model/inv_21/reshape_1/strided_slice/stack_1:output:0=InRFNet_Model/inv_21/reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,InRFNet_Model/inv_21/reshape_1/strided_slice?
.InRFNet_Model/inv_21/reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :820
.InRFNet_Model/inv_21/reshape_1/Reshape/shape/1?
.InRFNet_Model/inv_21/reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :820
.InRFNet_Model/inv_21/reshape_1/Reshape/shape/2?
.InRFNet_Model/inv_21/reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :20
.InRFNet_Model/inv_21/reshape_1/Reshape/shape/3?
.InRFNet_Model/inv_21/reshape_1/Reshape/shape/4Const*
_output_shapes
: *
dtype0*
value	B :20
.InRFNet_Model/inv_21/reshape_1/Reshape/shape/4?
.InRFNet_Model/inv_21/reshape_1/Reshape/shape/5Const*
_output_shapes
: *
dtype0*
value	B :20
.InRFNet_Model/inv_21/reshape_1/Reshape/shape/5?
,InRFNet_Model/inv_21/reshape_1/Reshape/shapePack5InRFNet_Model/inv_21/reshape_1/strided_slice:output:07InRFNet_Model/inv_21/reshape_1/Reshape/shape/1:output:07InRFNet_Model/inv_21/reshape_1/Reshape/shape/2:output:07InRFNet_Model/inv_21/reshape_1/Reshape/shape/3:output:07InRFNet_Model/inv_21/reshape_1/Reshape/shape/4:output:07InRFNet_Model/inv_21/reshape_1/Reshape/shape/5:output:0*
N*
T0*
_output_shapes
:2.
,InRFNet_Model/inv_21/reshape_1/Reshape/shape?
&InRFNet_Model/inv_21/reshape_1/ReshapeReshape2InRFNet_Model/inv_21/ExtractImagePatches:patches:05InRFNet_Model/inv_21/reshape_1/Reshape/shape:output:0*
T0*7
_output_shapes%
#:!?????????882(
&InRFNet_Model/inv_21/reshape_1/Reshape?
InRFNet_Model/inv_21/MulMul-InRFNet_Model/inv_21/reshape/Reshape:output:0/InRFNet_Model/inv_21/reshape_1/Reshape:output:0*
T0*7
_output_shapes%
#:!?????????882
InRFNet_Model/inv_21/Mul?
*InRFNet_Model/inv_21/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2,
*InRFNet_Model/inv_21/Sum/reduction_indices?
InRFNet_Model/inv_21/SumSumInRFNet_Model/inv_21/Mul:z:03InRFNet_Model/inv_21/Sum/reduction_indices:output:0*
T0*3
_output_shapes!
:?????????882
InRFNet_Model/inv_21/Sum?
$InRFNet_Model/inv_21/reshape_2/ShapeShape!InRFNet_Model/inv_21/Sum:output:0*
T0*
_output_shapes
:2&
$InRFNet_Model/inv_21/reshape_2/Shape?
2InRFNet_Model/inv_21/reshape_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 24
2InRFNet_Model/inv_21/reshape_2/strided_slice/stack?
4InRFNet_Model/inv_21/reshape_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:26
4InRFNet_Model/inv_21/reshape_2/strided_slice/stack_1?
4InRFNet_Model/inv_21/reshape_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4InRFNet_Model/inv_21/reshape_2/strided_slice/stack_2?
,InRFNet_Model/inv_21/reshape_2/strided_sliceStridedSlice-InRFNet_Model/inv_21/reshape_2/Shape:output:0;InRFNet_Model/inv_21/reshape_2/strided_slice/stack:output:0=InRFNet_Model/inv_21/reshape_2/strided_slice/stack_1:output:0=InRFNet_Model/inv_21/reshape_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,InRFNet_Model/inv_21/reshape_2/strided_slice?
.InRFNet_Model/inv_21/reshape_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :820
.InRFNet_Model/inv_21/reshape_2/Reshape/shape/1?
.InRFNet_Model/inv_21/reshape_2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :820
.InRFNet_Model/inv_21/reshape_2/Reshape/shape/2?
.InRFNet_Model/inv_21/reshape_2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :20
.InRFNet_Model/inv_21/reshape_2/Reshape/shape/3?
,InRFNet_Model/inv_21/reshape_2/Reshape/shapePack5InRFNet_Model/inv_21/reshape_2/strided_slice:output:07InRFNet_Model/inv_21/reshape_2/Reshape/shape/1:output:07InRFNet_Model/inv_21/reshape_2/Reshape/shape/2:output:07InRFNet_Model/inv_21/reshape_2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2.
,InRFNet_Model/inv_21/reshape_2/Reshape/shape?
&InRFNet_Model/inv_21/reshape_2/ReshapeReshape!InRFNet_Model/inv_21/Sum:output:05InRFNet_Model/inv_21/reshape_2/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????882(
&InRFNet_Model/inv_21/reshape_2/Reshape?
InRFNet_Model/re_lu_1/ReluRelu/InRFNet_Model/inv_21/reshape_2/Reshape:output:0*
T0*/
_output_shapes
:?????????882
InRFNet_Model/re_lu_1/Relu?
InRFNet_Model/re_lu_2/ReluRelu/InRFNet_Model/inv_22/reshape_2/Reshape:output:0*
T0*/
_output_shapes
:?????????882
InRFNet_Model/re_lu_2/Relu?
%InRFNet_Model/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2'
%InRFNet_Model/concatenate/concat/axis?
 InRFNet_Model/concatenate/concatConcatV2(InRFNet_Model/re_lu_1/Relu:activations:0(InRFNet_Model/re_lu_2/Relu:activations:0.InRFNet_Model/concatenate/concat/axis:output:0*
N*
T0*/
_output_shapes
:?????????882"
 InRFNet_Model/concatenate/concat?
%InRFNet_Model/max_pooling2d_1/MaxPoolMaxPool)InRFNet_Model/concatenate/concat:output:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2'
%InRFNet_Model/max_pooling2d_1/MaxPool?
InRFNet_Model/inv_3/IdentityIdentity.InRFNet_Model/max_pooling2d_1/MaxPool:output:0*
T0*/
_output_shapes
:?????????2
InRFNet_Model/inv_3/Identity?
;InRFNet_Model/inv_3/sequential/conv2d/Conv2D/ReadVariableOpReadVariableOpDinrfnet_model_inv_3_sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02=
;InRFNet_Model/inv_3/sequential/conv2d/Conv2D/ReadVariableOp?
,InRFNet_Model/inv_3/sequential/conv2d/Conv2DConv2D%InRFNet_Model/inv_3/Identity:output:0CInRFNet_Model/inv_3/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2.
,InRFNet_Model/inv_3/sequential/conv2d/Conv2D?
<InRFNet_Model/inv_3/sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOpEinrfnet_model_inv_3_sequential_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02>
<InRFNet_Model/inv_3/sequential/conv2d/BiasAdd/ReadVariableOp?
-InRFNet_Model/inv_3/sequential/conv2d/BiasAddBiasAdd5InRFNet_Model/inv_3/sequential/conv2d/Conv2D:output:0DInRFNet_Model/inv_3/sequential/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2/
-InRFNet_Model/inv_3/sequential/conv2d/BiasAdd?
AInRFNet_Model/inv_3/sequential/batch_normalization/ReadVariableOpReadVariableOpJinrfnet_model_inv_3_sequential_batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype02C
AInRFNet_Model/inv_3/sequential/batch_normalization/ReadVariableOp?
CInRFNet_Model/inv_3/sequential/batch_normalization/ReadVariableOp_1ReadVariableOpLinrfnet_model_inv_3_sequential_batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype02E
CInRFNet_Model/inv_3/sequential/batch_normalization/ReadVariableOp_1?
RInRFNet_Model/inv_3/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp[inrfnet_model_inv_3_sequential_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02T
RInRFNet_Model/inv_3/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp?
TInRFNet_Model/inv_3/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]inrfnet_model_inv_3_sequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02V
TInRFNet_Model/inv_3/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?
CInRFNet_Model/inv_3/sequential/batch_normalization/FusedBatchNormV3FusedBatchNormV36InRFNet_Model/inv_3/sequential/conv2d/BiasAdd:output:0IInRFNet_Model/inv_3/sequential/batch_normalization/ReadVariableOp:value:0KInRFNet_Model/inv_3/sequential/batch_normalization/ReadVariableOp_1:value:0ZInRFNet_Model/inv_3/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0\InRFNet_Model/inv_3/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
is_training( 2E
CInRFNet_Model/inv_3/sequential/batch_normalization/FusedBatchNormV3?
)InRFNet_Model/inv_3/sequential/re_lu/ReluReluGInRFNet_Model/inv_3/sequential/batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????2+
)InRFNet_Model/inv_3/sequential/re_lu/Relu?
=InRFNet_Model/inv_3/sequential/conv2d_1/Conv2D/ReadVariableOpReadVariableOpFinrfnet_model_inv_3_sequential_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:	*
dtype02?
=InRFNet_Model/inv_3/sequential/conv2d_1/Conv2D/ReadVariableOp?
.InRFNet_Model/inv_3/sequential/conv2d_1/Conv2DConv2D7InRFNet_Model/inv_3/sequential/re_lu/Relu:activations:0EInRFNet_Model/inv_3/sequential/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????	*
paddingVALID*
strides
20
.InRFNet_Model/inv_3/sequential/conv2d_1/Conv2D?
>InRFNet_Model/inv_3/sequential/conv2d_1/BiasAdd/ReadVariableOpReadVariableOpGinrfnet_model_inv_3_sequential_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02@
>InRFNet_Model/inv_3/sequential/conv2d_1/BiasAdd/ReadVariableOp?
/InRFNet_Model/inv_3/sequential/conv2d_1/BiasAddBiasAdd7InRFNet_Model/inv_3/sequential/conv2d_1/Conv2D:output:0FInRFNet_Model/inv_3/sequential/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????	21
/InRFNet_Model/inv_3/sequential/conv2d_1/BiasAdd?
!InRFNet_Model/inv_3/reshape/ShapeShape8InRFNet_Model/inv_3/sequential/conv2d_1/BiasAdd:output:0*
T0*
_output_shapes
:2#
!InRFNet_Model/inv_3/reshape/Shape?
/InRFNet_Model/inv_3/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/InRFNet_Model/inv_3/reshape/strided_slice/stack?
1InRFNet_Model/inv_3/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1InRFNet_Model/inv_3/reshape/strided_slice/stack_1?
1InRFNet_Model/inv_3/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1InRFNet_Model/inv_3/reshape/strided_slice/stack_2?
)InRFNet_Model/inv_3/reshape/strided_sliceStridedSlice*InRFNet_Model/inv_3/reshape/Shape:output:08InRFNet_Model/inv_3/reshape/strided_slice/stack:output:0:InRFNet_Model/inv_3/reshape/strided_slice/stack_1:output:0:InRFNet_Model/inv_3/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)InRFNet_Model/inv_3/reshape/strided_slice?
+InRFNet_Model/inv_3/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2-
+InRFNet_Model/inv_3/reshape/Reshape/shape/1?
+InRFNet_Model/inv_3/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2-
+InRFNet_Model/inv_3/reshape/Reshape/shape/2?
+InRFNet_Model/inv_3/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :	2-
+InRFNet_Model/inv_3/reshape/Reshape/shape/3?
+InRFNet_Model/inv_3/reshape/Reshape/shape/4Const*
_output_shapes
: *
dtype0*
value	B :2-
+InRFNet_Model/inv_3/reshape/Reshape/shape/4?
+InRFNet_Model/inv_3/reshape/Reshape/shape/5Const*
_output_shapes
: *
dtype0*
value	B :2-
+InRFNet_Model/inv_3/reshape/Reshape/shape/5?
)InRFNet_Model/inv_3/reshape/Reshape/shapePack2InRFNet_Model/inv_3/reshape/strided_slice:output:04InRFNet_Model/inv_3/reshape/Reshape/shape/1:output:04InRFNet_Model/inv_3/reshape/Reshape/shape/2:output:04InRFNet_Model/inv_3/reshape/Reshape/shape/3:output:04InRFNet_Model/inv_3/reshape/Reshape/shape/4:output:04InRFNet_Model/inv_3/reshape/Reshape/shape/5:output:0*
N*
T0*
_output_shapes
:2+
)InRFNet_Model/inv_3/reshape/Reshape/shape?
#InRFNet_Model/inv_3/reshape/ReshapeReshape8InRFNet_Model/inv_3/sequential/conv2d_1/BiasAdd:output:02InRFNet_Model/inv_3/reshape/Reshape/shape:output:0*
T0*7
_output_shapes%
#:!?????????	2%
#InRFNet_Model/inv_3/reshape/Reshape?
'InRFNet_Model/inv_3/ExtractImagePatchesExtractImagePatches.InRFNet_Model/max_pooling2d_1/MaxPool:output:0*
T0*/
_output_shapes
:?????????6*
ksizes
*
paddingSAME*
rates
*
strides
2)
'InRFNet_Model/inv_3/ExtractImagePatches?
#InRFNet_Model/inv_3/reshape_1/ShapeShape1InRFNet_Model/inv_3/ExtractImagePatches:patches:0*
T0*
_output_shapes
:2%
#InRFNet_Model/inv_3/reshape_1/Shape?
1InRFNet_Model/inv_3/reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1InRFNet_Model/inv_3/reshape_1/strided_slice/stack?
3InRFNet_Model/inv_3/reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3InRFNet_Model/inv_3/reshape_1/strided_slice/stack_1?
3InRFNet_Model/inv_3/reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3InRFNet_Model/inv_3/reshape_1/strided_slice/stack_2?
+InRFNet_Model/inv_3/reshape_1/strided_sliceStridedSlice,InRFNet_Model/inv_3/reshape_1/Shape:output:0:InRFNet_Model/inv_3/reshape_1/strided_slice/stack:output:0<InRFNet_Model/inv_3/reshape_1/strided_slice/stack_1:output:0<InRFNet_Model/inv_3/reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+InRFNet_Model/inv_3/reshape_1/strided_slice?
-InRFNet_Model/inv_3/reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2/
-InRFNet_Model/inv_3/reshape_1/Reshape/shape/1?
-InRFNet_Model/inv_3/reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2/
-InRFNet_Model/inv_3/reshape_1/Reshape/shape/2?
-InRFNet_Model/inv_3/reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :	2/
-InRFNet_Model/inv_3/reshape_1/Reshape/shape/3?
-InRFNet_Model/inv_3/reshape_1/Reshape/shape/4Const*
_output_shapes
: *
dtype0*
value	B :2/
-InRFNet_Model/inv_3/reshape_1/Reshape/shape/4?
-InRFNet_Model/inv_3/reshape_1/Reshape/shape/5Const*
_output_shapes
: *
dtype0*
value	B :2/
-InRFNet_Model/inv_3/reshape_1/Reshape/shape/5?
+InRFNet_Model/inv_3/reshape_1/Reshape/shapePack4InRFNet_Model/inv_3/reshape_1/strided_slice:output:06InRFNet_Model/inv_3/reshape_1/Reshape/shape/1:output:06InRFNet_Model/inv_3/reshape_1/Reshape/shape/2:output:06InRFNet_Model/inv_3/reshape_1/Reshape/shape/3:output:06InRFNet_Model/inv_3/reshape_1/Reshape/shape/4:output:06InRFNet_Model/inv_3/reshape_1/Reshape/shape/5:output:0*
N*
T0*
_output_shapes
:2-
+InRFNet_Model/inv_3/reshape_1/Reshape/shape?
%InRFNet_Model/inv_3/reshape_1/ReshapeReshape1InRFNet_Model/inv_3/ExtractImagePatches:patches:04InRFNet_Model/inv_3/reshape_1/Reshape/shape:output:0*
T0*7
_output_shapes%
#:!?????????	2'
%InRFNet_Model/inv_3/reshape_1/Reshape?
InRFNet_Model/inv_3/MulMul,InRFNet_Model/inv_3/reshape/Reshape:output:0.InRFNet_Model/inv_3/reshape_1/Reshape:output:0*
T0*7
_output_shapes%
#:!?????????	2
InRFNet_Model/inv_3/Mul?
)InRFNet_Model/inv_3/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2+
)InRFNet_Model/inv_3/Sum/reduction_indices?
InRFNet_Model/inv_3/SumSumInRFNet_Model/inv_3/Mul:z:02InRFNet_Model/inv_3/Sum/reduction_indices:output:0*
T0*3
_output_shapes!
:?????????2
InRFNet_Model/inv_3/Sum?
#InRFNet_Model/inv_3/reshape_2/ShapeShape InRFNet_Model/inv_3/Sum:output:0*
T0*
_output_shapes
:2%
#InRFNet_Model/inv_3/reshape_2/Shape?
1InRFNet_Model/inv_3/reshape_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 23
1InRFNet_Model/inv_3/reshape_2/strided_slice/stack?
3InRFNet_Model/inv_3/reshape_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3InRFNet_Model/inv_3/reshape_2/strided_slice/stack_1?
3InRFNet_Model/inv_3/reshape_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3InRFNet_Model/inv_3/reshape_2/strided_slice/stack_2?
+InRFNet_Model/inv_3/reshape_2/strided_sliceStridedSlice,InRFNet_Model/inv_3/reshape_2/Shape:output:0:InRFNet_Model/inv_3/reshape_2/strided_slice/stack:output:0<InRFNet_Model/inv_3/reshape_2/strided_slice/stack_1:output:0<InRFNet_Model/inv_3/reshape_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2-
+InRFNet_Model/inv_3/reshape_2/strided_slice?
-InRFNet_Model/inv_3/reshape_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2/
-InRFNet_Model/inv_3/reshape_2/Reshape/shape/1?
-InRFNet_Model/inv_3/reshape_2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2/
-InRFNet_Model/inv_3/reshape_2/Reshape/shape/2?
-InRFNet_Model/inv_3/reshape_2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2/
-InRFNet_Model/inv_3/reshape_2/Reshape/shape/3?
+InRFNet_Model/inv_3/reshape_2/Reshape/shapePack4InRFNet_Model/inv_3/reshape_2/strided_slice:output:06InRFNet_Model/inv_3/reshape_2/Reshape/shape/1:output:06InRFNet_Model/inv_3/reshape_2/Reshape/shape/2:output:06InRFNet_Model/inv_3/reshape_2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2-
+InRFNet_Model/inv_3/reshape_2/Reshape/shape?
%InRFNet_Model/inv_3/reshape_2/ReshapeReshape InRFNet_Model/inv_3/Sum:output:04InRFNet_Model/inv_3/reshape_2/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????2'
%InRFNet_Model/inv_3/reshape_2/Reshape?
InRFNet_Model/re_lu_3/ReluRelu.InRFNet_Model/inv_3/reshape_2/Reshape:output:0*
T0*/
_output_shapes
:?????????2
InRFNet_Model/re_lu_3/Relu?
InRFNet_Model/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????`  2
InRFNet_Model/flatten/Const?
InRFNet_Model/flatten/ReshapeReshape(InRFNet_Model/re_lu_3/Relu:activations:0$InRFNet_Model/flatten/Const:output:0*
T0*(
_output_shapes
:??????????$2
InRFNet_Model/flatten/Reshape?
)InRFNet_Model/dense/MatMul/ReadVariableOpReadVariableOp2inrfnet_model_dense_matmul_readvariableop_resource*
_output_shapes
:	?$@*
dtype02+
)InRFNet_Model/dense/MatMul/ReadVariableOp?
InRFNet_Model/dense/MatMulMatMul&InRFNet_Model/flatten/Reshape:output:01InRFNet_Model/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
InRFNet_Model/dense/MatMul?
*InRFNet_Model/dense/BiasAdd/ReadVariableOpReadVariableOp3inrfnet_model_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02,
*InRFNet_Model/dense/BiasAdd/ReadVariableOp?
InRFNet_Model/dense/BiasAddBiasAdd$InRFNet_Model/dense/MatMul:product:02InRFNet_Model/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
InRFNet_Model/dense/BiasAdd?
InRFNet_Model/dense/ReluRelu$InRFNet_Model/dense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
InRFNet_Model/dense/Relu?
+InRFNet_Model/dense_1/MatMul/ReadVariableOpReadVariableOp4inrfnet_model_dense_1_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02-
+InRFNet_Model/dense_1/MatMul/ReadVariableOp?
InRFNet_Model/dense_1/MatMulMatMul&InRFNet_Model/dense/Relu:activations:03InRFNet_Model/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
InRFNet_Model/dense_1/MatMul?
,InRFNet_Model/dense_1/BiasAdd/ReadVariableOpReadVariableOp5inrfnet_model_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,InRFNet_Model/dense_1/BiasAdd/ReadVariableOp?
InRFNet_Model/dense_1/BiasAddBiasAdd&InRFNet_Model/dense_1/MatMul:product:04InRFNet_Model/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
InRFNet_Model/dense_1/BiasAdd?
InRFNet_Model/dense_1/SoftmaxSoftmax&InRFNet_Model/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
InRFNet_Model/dense_1/Softmax?
IdentityIdentity'InRFNet_Model/dense_1/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp+^InRFNet_Model/dense/BiasAdd/ReadVariableOp*^InRFNet_Model/dense/MatMul/ReadVariableOp-^InRFNet_Model/dense_1/BiasAdd/ReadVariableOp,^InRFNet_Model/dense_1/MatMul/ReadVariableOpS^InRFNet_Model/inv_1/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpU^InRFNet_Model/inv_1/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1B^InRFNet_Model/inv_1/sequential/batch_normalization/ReadVariableOpD^InRFNet_Model/inv_1/sequential/batch_normalization/ReadVariableOp_1=^InRFNet_Model/inv_1/sequential/conv2d/BiasAdd/ReadVariableOp<^InRFNet_Model/inv_1/sequential/conv2d/Conv2D/ReadVariableOp?^InRFNet_Model/inv_1/sequential/conv2d_1/BiasAdd/ReadVariableOp>^InRFNet_Model/inv_1/sequential/conv2d_1/Conv2D/ReadVariableOpT^InRFNet_Model/inv_21/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpV^InRFNet_Model/inv_21/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1C^InRFNet_Model/inv_21/sequential/batch_normalization/ReadVariableOpE^InRFNet_Model/inv_21/sequential/batch_normalization/ReadVariableOp_1>^InRFNet_Model/inv_21/sequential/conv2d/BiasAdd/ReadVariableOp=^InRFNet_Model/inv_21/sequential/conv2d/Conv2D/ReadVariableOp@^InRFNet_Model/inv_21/sequential/conv2d_1/BiasAdd/ReadVariableOp?^InRFNet_Model/inv_21/sequential/conv2d_1/Conv2D/ReadVariableOpT^InRFNet_Model/inv_22/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpV^InRFNet_Model/inv_22/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1C^InRFNet_Model/inv_22/sequential/batch_normalization/ReadVariableOpE^InRFNet_Model/inv_22/sequential/batch_normalization/ReadVariableOp_1>^InRFNet_Model/inv_22/sequential/conv2d/BiasAdd/ReadVariableOp=^InRFNet_Model/inv_22/sequential/conv2d/Conv2D/ReadVariableOp@^InRFNet_Model/inv_22/sequential/conv2d_1/BiasAdd/ReadVariableOp?^InRFNet_Model/inv_22/sequential/conv2d_1/Conv2D/ReadVariableOpS^InRFNet_Model/inv_3/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpU^InRFNet_Model/inv_3/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1B^InRFNet_Model/inv_3/sequential/batch_normalization/ReadVariableOpD^InRFNet_Model/inv_3/sequential/batch_normalization/ReadVariableOp_1=^InRFNet_Model/inv_3/sequential/conv2d/BiasAdd/ReadVariableOp<^InRFNet_Model/inv_3/sequential/conv2d/Conv2D/ReadVariableOp?^InRFNet_Model/inv_3/sequential/conv2d_1/BiasAdd/ReadVariableOp>^InRFNet_Model/inv_3/sequential/conv2d_1/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*x
_input_shapesg
e:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2X
*InRFNet_Model/dense/BiasAdd/ReadVariableOp*InRFNet_Model/dense/BiasAdd/ReadVariableOp2V
)InRFNet_Model/dense/MatMul/ReadVariableOp)InRFNet_Model/dense/MatMul/ReadVariableOp2\
,InRFNet_Model/dense_1/BiasAdd/ReadVariableOp,InRFNet_Model/dense_1/BiasAdd/ReadVariableOp2Z
+InRFNet_Model/dense_1/MatMul/ReadVariableOp+InRFNet_Model/dense_1/MatMul/ReadVariableOp2?
RInRFNet_Model/inv_1/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpRInRFNet_Model/inv_1/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp2?
TInRFNet_Model/inv_1/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1TInRFNet_Model/inv_1/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_12?
AInRFNet_Model/inv_1/sequential/batch_normalization/ReadVariableOpAInRFNet_Model/inv_1/sequential/batch_normalization/ReadVariableOp2?
CInRFNet_Model/inv_1/sequential/batch_normalization/ReadVariableOp_1CInRFNet_Model/inv_1/sequential/batch_normalization/ReadVariableOp_12|
<InRFNet_Model/inv_1/sequential/conv2d/BiasAdd/ReadVariableOp<InRFNet_Model/inv_1/sequential/conv2d/BiasAdd/ReadVariableOp2z
;InRFNet_Model/inv_1/sequential/conv2d/Conv2D/ReadVariableOp;InRFNet_Model/inv_1/sequential/conv2d/Conv2D/ReadVariableOp2?
>InRFNet_Model/inv_1/sequential/conv2d_1/BiasAdd/ReadVariableOp>InRFNet_Model/inv_1/sequential/conv2d_1/BiasAdd/ReadVariableOp2~
=InRFNet_Model/inv_1/sequential/conv2d_1/Conv2D/ReadVariableOp=InRFNet_Model/inv_1/sequential/conv2d_1/Conv2D/ReadVariableOp2?
SInRFNet_Model/inv_21/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpSInRFNet_Model/inv_21/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp2?
UInRFNet_Model/inv_21/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1UInRFNet_Model/inv_21/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_12?
BInRFNet_Model/inv_21/sequential/batch_normalization/ReadVariableOpBInRFNet_Model/inv_21/sequential/batch_normalization/ReadVariableOp2?
DInRFNet_Model/inv_21/sequential/batch_normalization/ReadVariableOp_1DInRFNet_Model/inv_21/sequential/batch_normalization/ReadVariableOp_12~
=InRFNet_Model/inv_21/sequential/conv2d/BiasAdd/ReadVariableOp=InRFNet_Model/inv_21/sequential/conv2d/BiasAdd/ReadVariableOp2|
<InRFNet_Model/inv_21/sequential/conv2d/Conv2D/ReadVariableOp<InRFNet_Model/inv_21/sequential/conv2d/Conv2D/ReadVariableOp2?
?InRFNet_Model/inv_21/sequential/conv2d_1/BiasAdd/ReadVariableOp?InRFNet_Model/inv_21/sequential/conv2d_1/BiasAdd/ReadVariableOp2?
>InRFNet_Model/inv_21/sequential/conv2d_1/Conv2D/ReadVariableOp>InRFNet_Model/inv_21/sequential/conv2d_1/Conv2D/ReadVariableOp2?
SInRFNet_Model/inv_22/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpSInRFNet_Model/inv_22/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp2?
UInRFNet_Model/inv_22/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1UInRFNet_Model/inv_22/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_12?
BInRFNet_Model/inv_22/sequential/batch_normalization/ReadVariableOpBInRFNet_Model/inv_22/sequential/batch_normalization/ReadVariableOp2?
DInRFNet_Model/inv_22/sequential/batch_normalization/ReadVariableOp_1DInRFNet_Model/inv_22/sequential/batch_normalization/ReadVariableOp_12~
=InRFNet_Model/inv_22/sequential/conv2d/BiasAdd/ReadVariableOp=InRFNet_Model/inv_22/sequential/conv2d/BiasAdd/ReadVariableOp2|
<InRFNet_Model/inv_22/sequential/conv2d/Conv2D/ReadVariableOp<InRFNet_Model/inv_22/sequential/conv2d/Conv2D/ReadVariableOp2?
?InRFNet_Model/inv_22/sequential/conv2d_1/BiasAdd/ReadVariableOp?InRFNet_Model/inv_22/sequential/conv2d_1/BiasAdd/ReadVariableOp2?
>InRFNet_Model/inv_22/sequential/conv2d_1/Conv2D/ReadVariableOp>InRFNet_Model/inv_22/sequential/conv2d_1/Conv2D/ReadVariableOp2?
RInRFNet_Model/inv_3/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpRInRFNet_Model/inv_3/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp2?
TInRFNet_Model/inv_3/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1TInRFNet_Model/inv_3/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_12?
AInRFNet_Model/inv_3/sequential/batch_normalization/ReadVariableOpAInRFNet_Model/inv_3/sequential/batch_normalization/ReadVariableOp2?
CInRFNet_Model/inv_3/sequential/batch_normalization/ReadVariableOp_1CInRFNet_Model/inv_3/sequential/batch_normalization/ReadVariableOp_12|
<InRFNet_Model/inv_3/sequential/conv2d/BiasAdd/ReadVariableOp<InRFNet_Model/inv_3/sequential/conv2d/BiasAdd/ReadVariableOp2z
;InRFNet_Model/inv_3/sequential/conv2d/Conv2D/ReadVariableOp;InRFNet_Model/inv_3/sequential/conv2d/Conv2D/ReadVariableOp2?
>InRFNet_Model/inv_3/sequential/conv2d_1/BiasAdd/ReadVariableOp>InRFNet_Model/inv_3/sequential/conv2d_1/BiasAdd/ReadVariableOp2~
=InRFNet_Model/inv_3/sequential/conv2d_1/Conv2D/ReadVariableOp=InRFNet_Model/inv_3/sequential/conv2d_1/Conv2D/ReadVariableOp:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
?C
?
I__inference_InRFNet_Model_layer_call_and_return_conditional_losses_304082

inputs&
inv_1_303709:
inv_1_303711:
inv_1_303713:
inv_1_303715:
inv_1_303717:
inv_1_303719:&
inv_1_303721:	
inv_1_303723:	'
inv_22_303810:
inv_22_303812:
inv_22_303814:
inv_22_303816:
inv_22_303818:
inv_22_303820:'
inv_22_303822:
inv_22_303824:'
inv_21_303898:
inv_21_303900:
inv_21_303902:
inv_21_303904:
inv_21_303906:
inv_21_303908:'
inv_21_303910:
inv_21_303912:&
inv_3_304015:
inv_3_304017:
inv_3_304019:
inv_3_304021:
inv_3_304023:
inv_3_304025:&
inv_3_304027:	
inv_3_304029:	
dense_304059:	?$@
dense_304061:@ 
dense_1_304076:@
dense_1_304078:
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?inv_1/StatefulPartitionedCall?inv_21/StatefulPartitionedCall?inv_22/StatefulPartitionedCall?inv_3/StatefulPartitionedCall?
inv_1/StatefulPartitionedCallStatefulPartitionedCallinputsinv_1_303709inv_1_303711inv_1_303713inv_1_303715inv_1_303717inv_1_303719inv_1_303721inv_1_303723*
Tin
2	*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_inv_1_layer_call_and_return_conditional_losses_3037082
inv_1/StatefulPartitionedCall?
re_lu/PartitionedCallPartitionedCall&inv_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_re_lu_layer_call_and_return_conditional_losses_3037312
re_lu/PartitionedCall?
max_pooling2d/PartitionedCallPartitionedCallre_lu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????pp* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_3037372
max_pooling2d/PartitionedCall?
inv_22/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0inv_22_303810inv_22_303812inv_22_303814inv_22_303816inv_22_303818inv_22_303820inv_22_303822inv_22_303824*
Tin
2	*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????88**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_inv_22_layer_call_and_return_conditional_losses_3038092 
inv_22/StatefulPartitionedCall?
inv_21/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0inv_21_303898inv_21_303900inv_21_303902inv_21_303904inv_21_303906inv_21_303908inv_21_303910inv_21_303912*
Tin
2	*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????88**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_inv_21_layer_call_and_return_conditional_losses_3038972 
inv_21/StatefulPartitionedCall?
re_lu_1/PartitionedCallPartitionedCall'inv_21/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????88* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_re_lu_1_layer_call_and_return_conditional_losses_3039202
re_lu_1/PartitionedCall?
re_lu_2/PartitionedCallPartitionedCall'inv_22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????88* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_re_lu_2_layer_call_and_return_conditional_losses_3039272
re_lu_2/PartitionedCall?
concatenate/PartitionedCallPartitionedCall re_lu_1/PartitionedCall:output:0 re_lu_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????88* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_3039362
concatenate/PartitionedCall?
max_pooling2d_1/PartitionedCallPartitionedCall$concatenate/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_3039422!
max_pooling2d_1/PartitionedCall?
inv_3/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0inv_3_304015inv_3_304017inv_3_304019inv_3_304021inv_3_304023inv_3_304025inv_3_304027inv_3_304029*
Tin
2	*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_inv_3_layer_call_and_return_conditional_losses_3040142
inv_3/StatefulPartitionedCall?
re_lu_3/PartitionedCallPartitionedCall&inv_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_re_lu_3_layer_call_and_return_conditional_losses_3040372
re_lu_3/PartitionedCall?
flatten/PartitionedCallPartitionedCall re_lu_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????$* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_3040452
flatten/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_304059dense_304061*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_3040582
dense/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_304076dense_1_304078*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_3040752!
dense_1/StatefulPartitionedCall?
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall^inv_1/StatefulPartitionedCall^inv_21/StatefulPartitionedCall^inv_22/StatefulPartitionedCall^inv_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*x
_input_shapesg
e:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2>
inv_1/StatefulPartitionedCallinv_1/StatefulPartitionedCall2@
inv_21/StatefulPartitionedCallinv_21/StatefulPartitionedCall2@
inv_22/StatefulPartitionedCallinv_22/StatefulPartitionedCall2>
inv_3/StatefulPartitionedCallinv_3/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
D
(__inference_flatten_layer_call_fn_306832

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????$* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_3040452
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????$2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
]
A__inference_re_lu_layer_call_and_return_conditional_losses_302496

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????882
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????882

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????88:W S
/
_output_shapes
:?????????88
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_layer_call_and_return_conditional_losses_303029

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????88:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????882

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????88: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????88
 
_user_specified_nameinputs
?

?
+__inference_sequential_layer_call_fn_302223
conv2d_input!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5:	
	unknown_6:	
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????	*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_3021832
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????	2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:???????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:_ [
1
_output_shapes
:???????????
&
_user_specified_nameconv2d_input
?h
?	
A__inference_inv_3_layer_call_and_return_conditional_losses_306769
xJ
0sequential_conv2d_conv2d_readvariableop_resource:?
1sequential_conv2d_biasadd_readvariableop_resource:D
6sequential_batch_normalization_readvariableop_resource:F
8sequential_batch_normalization_readvariableop_1_resource:U
Gsequential_batch_normalization_fusedbatchnormv3_readvariableop_resource:W
Isequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource:L
2sequential_conv2d_1_conv2d_readvariableop_resource:	A
3sequential_conv2d_1_biasadd_readvariableop_resource:	

identity_1??-sequential/batch_normalization/AssignNewValue?/sequential/batch_normalization/AssignNewValue_1?>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp?@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?-sequential/batch_normalization/ReadVariableOp?/sequential/batch_normalization/ReadVariableOp_1?(sequential/conv2d/BiasAdd/ReadVariableOp?'sequential/conv2d/Conv2D/ReadVariableOp?*sequential/conv2d_1/BiasAdd/ReadVariableOp?)sequential/conv2d_1/Conv2D/ReadVariableOp]
IdentityIdentityx*
T0*/
_output_shapes
:?????????2

Identity?
'sequential/conv2d/Conv2D/ReadVariableOpReadVariableOp0sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02)
'sequential/conv2d/Conv2D/ReadVariableOp?
sequential/conv2d/Conv2DConv2DIdentity:output:0/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
sequential/conv2d/Conv2D?
(sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOp1sequential_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(sequential/conv2d/BiasAdd/ReadVariableOp?
sequential/conv2d/BiasAddBiasAdd!sequential/conv2d/Conv2D:output:00sequential/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
sequential/conv2d/BiasAdd?
-sequential/batch_normalization/ReadVariableOpReadVariableOp6sequential_batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential/batch_normalization/ReadVariableOp?
/sequential/batch_normalization/ReadVariableOp_1ReadVariableOp8sequential_batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype021
/sequential/batch_normalization/ReadVariableOp_1?
>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpGsequential_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02@
>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp?
@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIsequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02B
@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?
/sequential/batch_normalization/FusedBatchNormV3FusedBatchNormV3"sequential/conv2d/BiasAdd:output:05sequential/batch_normalization/ReadVariableOp:value:07sequential/batch_normalization/ReadVariableOp_1:value:0Fsequential/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Hsequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<21
/sequential/batch_normalization/FusedBatchNormV3?
-sequential/batch_normalization/AssignNewValueAssignVariableOpGsequential_batch_normalization_fusedbatchnormv3_readvariableop_resource<sequential/batch_normalization/FusedBatchNormV3:batch_mean:0?^sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02/
-sequential/batch_normalization/AssignNewValue?
/sequential/batch_normalization/AssignNewValue_1AssignVariableOpIsequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource@sequential/batch_normalization/FusedBatchNormV3:batch_variance:0A^sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype021
/sequential/batch_normalization/AssignNewValue_1?
sequential/re_lu/ReluRelu3sequential/batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????2
sequential/re_lu/Relu?
)sequential/conv2d_1/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:	*
dtype02+
)sequential/conv2d_1/Conv2D/ReadVariableOp?
sequential/conv2d_1/Conv2DConv2D#sequential/re_lu/Relu:activations:01sequential/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????	*
paddingVALID*
strides
2
sequential/conv2d_1/Conv2D?
*sequential/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02,
*sequential/conv2d_1/BiasAdd/ReadVariableOp?
sequential/conv2d_1/BiasAddBiasAdd#sequential/conv2d_1/Conv2D:output:02sequential/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????	2
sequential/conv2d_1/BiasAddr
reshape/ShapeShape$sequential/conv2d_1/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape/Shape?
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack?
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1?
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2?
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slicet
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/2t
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :	2
reshape/Reshape/shape/3t
reshape/Reshape/shape/4Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/4t
reshape/Reshape/shape/5Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/5?
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0 reshape/Reshape/shape/4:output:0 reshape/Reshape/shape/5:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape?
reshape/ReshapeReshape$sequential/conv2d_1/BiasAdd:output:0reshape/Reshape/shape:output:0*
T0*7
_output_shapes%
#:!?????????	2
reshape/Reshape?
ExtractImagePatchesExtractImagePatchesx*
T0*/
_output_shapes
:?????????6*
ksizes
*
paddingSAME*
rates
*
strides
2
ExtractImagePatcheso
reshape_1/ShapeShapeExtractImagePatches:patches:0*
T0*
_output_shapes
:2
reshape_1/Shape?
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_1/strided_slice/stack?
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_1?
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_2?
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_1/strided_slicex
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/1x
reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/2x
reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :	2
reshape_1/Reshape/shape/3x
reshape_1/Reshape/shape/4Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/4x
reshape_1/Reshape/shape/5Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/5?
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0"reshape_1/Reshape/shape/2:output:0"reshape_1/Reshape/shape/3:output:0"reshape_1/Reshape/shape/4:output:0"reshape_1/Reshape/shape/5:output:0*
N*
T0*
_output_shapes
:2
reshape_1/Reshape/shape?
reshape_1/ReshapeReshapeExtractImagePatches:patches:0 reshape_1/Reshape/shape:output:0*
T0*7
_output_shapes%
#:!?????????	2
reshape_1/Reshape?
MulMulreshape/Reshape:output:0reshape_1/Reshape:output:0*
T0*7
_output_shapes%
#:!?????????	2
Mulp
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indicesx
SumSumMul:z:0Sum/reduction_indices:output:0*
T0*3
_output_shapes!
:?????????2
Sum^
reshape_2/ShapeShapeSum:output:0*
T0*
_output_shapes
:2
reshape_2/Shape?
reshape_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_2/strided_slice/stack?
reshape_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_2/strided_slice/stack_1?
reshape_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_2/strided_slice/stack_2?
reshape_2/strided_sliceStridedSlicereshape_2/Shape:output:0&reshape_2/strided_slice/stack:output:0(reshape_2/strided_slice/stack_1:output:0(reshape_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_2/strided_slicex
reshape_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_2/Reshape/shape/1x
reshape_2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_2/Reshape/shape/2x
reshape_2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_2/Reshape/shape/3?
reshape_2/Reshape/shapePack reshape_2/strided_slice:output:0"reshape_2/Reshape/shape/1:output:0"reshape_2/Reshape/shape/2:output:0"reshape_2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_2/Reshape/shape?
reshape_2/ReshapeReshapeSum:output:0 reshape_2/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????2
reshape_2/Reshape?

Identity_1Identityreshape_2/Reshape:output:0^NoOp*
T0*/
_output_shapes
:?????????2

Identity_1?
NoOpNoOp.^sequential/batch_normalization/AssignNewValue0^sequential/batch_normalization/AssignNewValue_1?^sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpA^sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1.^sequential/batch_normalization/ReadVariableOp0^sequential/batch_normalization/ReadVariableOp_1)^sequential/conv2d/BiasAdd/ReadVariableOp(^sequential/conv2d/Conv2D/ReadVariableOp+^sequential/conv2d_1/BiasAdd/ReadVariableOp*^sequential/conv2d_1/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????: : : : : : : : 2^
-sequential/batch_normalization/AssignNewValue-sequential/batch_normalization/AssignNewValue2b
/sequential/batch_normalization/AssignNewValue_1/sequential/batch_normalization/AssignNewValue_12?
>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp2?
@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_12^
-sequential/batch_normalization/ReadVariableOp-sequential/batch_normalization/ReadVariableOp2b
/sequential/batch_normalization/ReadVariableOp_1/sequential/batch_normalization/ReadVariableOp_12T
(sequential/conv2d/BiasAdd/ReadVariableOp(sequential/conv2d/BiasAdd/ReadVariableOp2R
'sequential/conv2d/Conv2D/ReadVariableOp'sequential/conv2d/Conv2D/ReadVariableOp2X
*sequential/conv2d_1/BiasAdd/ReadVariableOp*sequential/conv2d_1/BiasAdd/ReadVariableOp2V
)sequential/conv2d_1/Conv2D/ReadVariableOp)sequential/conv2d_1/Conv2D/ReadVariableOp:R N
/
_output_shapes
:?????????

_user_specified_namex
?(
?
F__inference_sequential_layer_call_and_return_conditional_losses_307017

inputs?
%conv2d_conv2d_readvariableop_resource:4
&conv2d_biasadd_readvariableop_resource:9
+batch_normalization_readvariableop_resource:;
-batch_normalization_readvariableop_1_resource:J
<batch_normalization_fusedbatchnormv3_readvariableop_resource:L
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource:A
'conv2d_1_conv2d_readvariableop_resource:6
(conv2d_1_biasadd_readvariableop_resource:
identity??3batch_normalization/FusedBatchNormV3/ReadVariableOp?5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?"batch_normalization/ReadVariableOp?$batch_normalization/ReadVariableOp_1?conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????88*
paddingVALID*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????882
conv2d/BiasAdd?
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype02$
"batch_normalization/ReadVariableOp?
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype02&
$batch_normalization/ReadVariableOp_1?
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOp?
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d/BiasAdd:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????88:::::*
epsilon%o?:*
is_training( 2&
$batch_normalization/FusedBatchNormV3?

re_lu/ReluRelu(batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????882

re_lu/Relu?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2Dre_lu/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????88*
paddingVALID*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????882
conv2d_1/BiasAdd|
IdentityIdentityconv2d_1/BiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????882

Identity?
NoOpNoOp4^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_1^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????88: : : : : : : : 2j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????88
 
_user_specified_nameinputs
?C
?
I__inference_InRFNet_Model_layer_call_and_return_conditional_losses_304845

inputs&
inv_1_304758:
inv_1_304760:
inv_1_304762:
inv_1_304764:
inv_1_304766:
inv_1_304768:&
inv_1_304770:	
inv_1_304772:	'
inv_22_304777:
inv_22_304779:
inv_22_304781:
inv_22_304783:
inv_22_304785:
inv_22_304787:'
inv_22_304789:
inv_22_304791:'
inv_21_304794:
inv_21_304796:
inv_21_304798:
inv_21_304800:
inv_21_304802:
inv_21_304804:'
inv_21_304806:
inv_21_304808:&
inv_3_304815:
inv_3_304817:
inv_3_304819:
inv_3_304821:
inv_3_304823:
inv_3_304825:&
inv_3_304827:	
inv_3_304829:	
dense_304834:	?$@
dense_304836:@ 
dense_1_304839:@
dense_1_304841:
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?inv_1/StatefulPartitionedCall?inv_21/StatefulPartitionedCall?inv_22/StatefulPartitionedCall?inv_3/StatefulPartitionedCall?
inv_1/StatefulPartitionedCallStatefulPartitionedCallinputsinv_1_304758inv_1_304760inv_1_304762inv_1_304764inv_1_304766inv_1_304768inv_1_304770inv_1_304772*
Tin
2	*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_inv_1_layer_call_and_return_conditional_losses_3046572
inv_1/StatefulPartitionedCall?
re_lu/PartitionedCallPartitionedCall&inv_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_re_lu_layer_call_and_return_conditional_losses_3037312
re_lu/PartitionedCall?
max_pooling2d/PartitionedCallPartitionedCallre_lu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????pp* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_3037372
max_pooling2d/PartitionedCall?
inv_22/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0inv_22_304777inv_22_304779inv_22_304781inv_22_304783inv_22_304785inv_22_304787inv_22_304789inv_22_304791*
Tin
2	*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????88*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_inv_22_layer_call_and_return_conditional_losses_3045332 
inv_22/StatefulPartitionedCall?
inv_21/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0inv_21_304794inv_21_304796inv_21_304798inv_21_304800inv_21_304802inv_21_304804inv_21_304806inv_21_304808*
Tin
2	*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????88*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_inv_21_layer_call_and_return_conditional_losses_3044202 
inv_21/StatefulPartitionedCall?
re_lu_1/PartitionedCallPartitionedCall'inv_21/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????88* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_re_lu_1_layer_call_and_return_conditional_losses_3039202
re_lu_1/PartitionedCall?
re_lu_2/PartitionedCallPartitionedCall'inv_22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????88* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_re_lu_2_layer_call_and_return_conditional_losses_3039272
re_lu_2/PartitionedCall?
concatenate/PartitionedCallPartitionedCall re_lu_1/PartitionedCall:output:0 re_lu_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????88* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_3039362
concatenate/PartitionedCall?
max_pooling2d_1/PartitionedCallPartitionedCall$concatenate/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_3039422!
max_pooling2d_1/PartitionedCall?
inv_3/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0inv_3_304815inv_3_304817inv_3_304819inv_3_304821inv_3_304823inv_3_304825inv_3_304827inv_3_304829*
Tin
2	*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_inv_3_layer_call_and_return_conditional_losses_3042832
inv_3/StatefulPartitionedCall?
re_lu_3/PartitionedCallPartitionedCall&inv_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_re_lu_3_layer_call_and_return_conditional_losses_3040372
re_lu_3/PartitionedCall?
flatten/PartitionedCallPartitionedCall re_lu_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????$* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_3040452
flatten/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_304834dense_304836*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_3040582
dense/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_304839dense_1_304841*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_3040752!
dense_1/StatefulPartitionedCall?
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall^inv_1/StatefulPartitionedCall^inv_21/StatefulPartitionedCall^inv_22/StatefulPartitionedCall^inv_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*x
_input_shapesg
e:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2>
inv_1/StatefulPartitionedCallinv_1/StatefulPartitionedCall2@
inv_21/StatefulPartitionedCallinv_21/StatefulPartitionedCall2@
inv_22/StatefulPartitionedCallinv_22/StatefulPartitionedCall2>
inv_3/StatefulPartitionedCallinv_3/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?(
?
F__inference_sequential_layer_call_and_return_conditional_losses_306903

inputs?
%conv2d_conv2d_readvariableop_resource:4
&conv2d_biasadd_readvariableop_resource:9
+batch_normalization_readvariableop_resource:;
-batch_normalization_readvariableop_1_resource:J
<batch_normalization_fusedbatchnormv3_readvariableop_resource:L
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource:A
'conv2d_1_conv2d_readvariableop_resource:	6
(conv2d_1_biasadd_readvariableop_resource:	
identity??3batch_normalization/FusedBatchNormV3/ReadVariableOp?5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?"batch_normalization/ReadVariableOp?$batch_normalization/ReadVariableOp_1?conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d/BiasAdd?
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype02$
"batch_normalization/ReadVariableOp?
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype02&
$batch_normalization/ReadVariableOp_1?
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOp?
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d/BiasAdd:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( 2&
$batch_normalization/FusedBatchNormV3?

re_lu/ReluRelu(batch_normalization/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????2

re_lu/Relu?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:	*
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2Dre_lu/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????	*
paddingVALID*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????	2
conv2d_1/BiasAdd~
IdentityIdentityconv2d_1/BiasAdd:output:0^NoOp*
T0*1
_output_shapes
:???????????	2

Identity?
NoOpNoOp4^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_1^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:???????????: : : : : : : : 2j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?`
?
B__inference_inv_21_layer_call_and_return_conditional_losses_303897
xJ
0sequential_conv2d_conv2d_readvariableop_resource:?
1sequential_conv2d_biasadd_readvariableop_resource:D
6sequential_batch_normalization_readvariableop_resource:F
8sequential_batch_normalization_readvariableop_1_resource:U
Gsequential_batch_normalization_fusedbatchnormv3_readvariableop_resource:W
Isequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource:L
2sequential_conv2d_1_conv2d_readvariableop_resource:A
3sequential_conv2d_1_biasadd_readvariableop_resource:
identity??>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp?@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?-sequential/batch_normalization/ReadVariableOp?/sequential/batch_normalization/ReadVariableOp_1?(sequential/conv2d/BiasAdd/ReadVariableOp?'sequential/conv2d/Conv2D/ReadVariableOp?*sequential/conv2d_1/BiasAdd/ReadVariableOp?)sequential/conv2d_1/Conv2D/ReadVariableOp?
average_pooling2d/AvgPoolAvgPoolx*
T0*/
_output_shapes
:?????????88*
ksize
*
paddingSAME*
strides
2
average_pooling2d/AvgPool?
'sequential/conv2d/Conv2D/ReadVariableOpReadVariableOp0sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02)
'sequential/conv2d/Conv2D/ReadVariableOp?
sequential/conv2d/Conv2DConv2D"average_pooling2d/AvgPool:output:0/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????88*
paddingVALID*
strides
2
sequential/conv2d/Conv2D?
(sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOp1sequential_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(sequential/conv2d/BiasAdd/ReadVariableOp?
sequential/conv2d/BiasAddBiasAdd!sequential/conv2d/Conv2D:output:00sequential/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????882
sequential/conv2d/BiasAdd?
-sequential/batch_normalization/ReadVariableOpReadVariableOp6sequential_batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential/batch_normalization/ReadVariableOp?
/sequential/batch_normalization/ReadVariableOp_1ReadVariableOp8sequential_batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype021
/sequential/batch_normalization/ReadVariableOp_1?
>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpGsequential_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02@
>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp?
@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIsequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02B
@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?
/sequential/batch_normalization/FusedBatchNormV3FusedBatchNormV3"sequential/conv2d/BiasAdd:output:05sequential/batch_normalization/ReadVariableOp:value:07sequential/batch_normalization/ReadVariableOp_1:value:0Fsequential/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Hsequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????88:::::*
epsilon%o?:*
is_training( 21
/sequential/batch_normalization/FusedBatchNormV3?
sequential/re_lu/ReluRelu3sequential/batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????882
sequential/re_lu/Relu?
)sequential/conv2d_1/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02+
)sequential/conv2d_1/Conv2D/ReadVariableOp?
sequential/conv2d_1/Conv2DConv2D#sequential/re_lu/Relu:activations:01sequential/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????88*
paddingVALID*
strides
2
sequential/conv2d_1/Conv2D?
*sequential/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*sequential/conv2d_1/BiasAdd/ReadVariableOp?
sequential/conv2d_1/BiasAddBiasAdd#sequential/conv2d_1/Conv2D:output:02sequential/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????882
sequential/conv2d_1/BiasAddr
reshape/ShapeShape$sequential/conv2d_1/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape/Shape?
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack?
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1?
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2?
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slicet
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :82
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :82
reshape/Reshape/shape/2t
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/3t
reshape/Reshape/shape/4Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/4t
reshape/Reshape/shape/5Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/5?
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0 reshape/Reshape/shape/4:output:0 reshape/Reshape/shape/5:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape?
reshape/ReshapeReshape$sequential/conv2d_1/BiasAdd:output:0reshape/Reshape/shape:output:0*
T0*7
_output_shapes%
#:!?????????882
reshape/Reshape?
ExtractImagePatchesExtractImagePatchesx*
T0*/
_output_shapes
:?????????88	*
ksizes
*
paddingSAME*
rates
*
strides
2
ExtractImagePatcheso
reshape_1/ShapeShapeExtractImagePatches:patches:0*
T0*
_output_shapes
:2
reshape_1/Shape?
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_1/strided_slice/stack?
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_1?
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_2?
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_1/strided_slicex
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :82
reshape_1/Reshape/shape/1x
reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :82
reshape_1/Reshape/shape/2x
reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/3x
reshape_1/Reshape/shape/4Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/4x
reshape_1/Reshape/shape/5Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/5?
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0"reshape_1/Reshape/shape/2:output:0"reshape_1/Reshape/shape/3:output:0"reshape_1/Reshape/shape/4:output:0"reshape_1/Reshape/shape/5:output:0*
N*
T0*
_output_shapes
:2
reshape_1/Reshape/shape?
reshape_1/ReshapeReshapeExtractImagePatches:patches:0 reshape_1/Reshape/shape:output:0*
T0*7
_output_shapes%
#:!?????????882
reshape_1/Reshape?
MulMulreshape/Reshape:output:0reshape_1/Reshape:output:0*
T0*7
_output_shapes%
#:!?????????882
Mulp
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indicesx
SumSumMul:z:0Sum/reduction_indices:output:0*
T0*3
_output_shapes!
:?????????882
Sum^
reshape_2/ShapeShapeSum:output:0*
T0*
_output_shapes
:2
reshape_2/Shape?
reshape_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_2/strided_slice/stack?
reshape_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_2/strided_slice/stack_1?
reshape_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_2/strided_slice/stack_2?
reshape_2/strided_sliceStridedSlicereshape_2/Shape:output:0&reshape_2/strided_slice/stack:output:0(reshape_2/strided_slice/stack_1:output:0(reshape_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_2/strided_slicex
reshape_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :82
reshape_2/Reshape/shape/1x
reshape_2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :82
reshape_2/Reshape/shape/2x
reshape_2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_2/Reshape/shape/3?
reshape_2/Reshape/shapePack reshape_2/strided_slice:output:0"reshape_2/Reshape/shape/1:output:0"reshape_2/Reshape/shape/2:output:0"reshape_2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_2/Reshape/shape?
reshape_2/ReshapeReshapeSum:output:0 reshape_2/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????882
reshape_2/Reshape}
IdentityIdentityreshape_2/Reshape:output:0^NoOp*
T0*/
_output_shapes
:?????????882

Identity?
NoOpNoOp?^sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpA^sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1.^sequential/batch_normalization/ReadVariableOp0^sequential/batch_normalization/ReadVariableOp_1)^sequential/conv2d/BiasAdd/ReadVariableOp(^sequential/conv2d/Conv2D/ReadVariableOp+^sequential/conv2d_1/BiasAdd/ReadVariableOp*^sequential/conv2d_1/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????pp: : : : : : : : 2?
>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp2?
@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_12^
-sequential/batch_normalization/ReadVariableOp-sequential/batch_normalization/ReadVariableOp2b
/sequential/batch_normalization/ReadVariableOp_1/sequential/batch_normalization/ReadVariableOp_12T
(sequential/conv2d/BiasAdd/ReadVariableOp(sequential/conv2d/BiasAdd/ReadVariableOp2R
'sequential/conv2d/Conv2D/ReadVariableOp'sequential/conv2d/Conv2D/ReadVariableOp2X
*sequential/conv2d_1/BiasAdd/ReadVariableOp*sequential/conv2d_1/BiasAdd/ReadVariableOp2V
)sequential/conv2d_1/Conv2D/ReadVariableOp)sequential/conv2d_1/Conv2D/ReadVariableOp:R N
/
_output_shapes
:?????????pp

_user_specified_namex
?	
?
4__inference_batch_normalization_layer_call_fn_307941

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_3032732
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?

?
+__inference_sequential_layer_call_fn_307090

inputs!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????88*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_3026512
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????882

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????88: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????88
 
_user_specified_nameinputs
?
?
A__inference_dense_layer_call_and_return_conditional_losses_306843

inputs1
matmul_readvariableop_resource:	?$@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?$@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????$: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????$
 
_user_specified_nameinputs
?
?
(__inference_dense_1_layer_call_fn_306872

inputs
unknown:@
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_3040752
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?i
?	
B__inference_inv_22_layer_call_and_return_conditional_losses_306534
xJ
0sequential_conv2d_conv2d_readvariableop_resource:?
1sequential_conv2d_biasadd_readvariableop_resource:D
6sequential_batch_normalization_readvariableop_resource:F
8sequential_batch_normalization_readvariableop_1_resource:U
Gsequential_batch_normalization_fusedbatchnormv3_readvariableop_resource:W
Isequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource:L
2sequential_conv2d_1_conv2d_readvariableop_resource:A
3sequential_conv2d_1_biasadd_readvariableop_resource:
identity??-sequential/batch_normalization/AssignNewValue?/sequential/batch_normalization/AssignNewValue_1?>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp?@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?-sequential/batch_normalization/ReadVariableOp?/sequential/batch_normalization/ReadVariableOp_1?(sequential/conv2d/BiasAdd/ReadVariableOp?'sequential/conv2d/Conv2D/ReadVariableOp?*sequential/conv2d_1/BiasAdd/ReadVariableOp?)sequential/conv2d_1/Conv2D/ReadVariableOp?
average_pooling2d/AvgPoolAvgPoolx*
T0*/
_output_shapes
:?????????88*
ksize
*
paddingSAME*
strides
2
average_pooling2d/AvgPool?
'sequential/conv2d/Conv2D/ReadVariableOpReadVariableOp0sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02)
'sequential/conv2d/Conv2D/ReadVariableOp?
sequential/conv2d/Conv2DConv2D"average_pooling2d/AvgPool:output:0/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????88*
paddingVALID*
strides
2
sequential/conv2d/Conv2D?
(sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOp1sequential_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(sequential/conv2d/BiasAdd/ReadVariableOp?
sequential/conv2d/BiasAddBiasAdd!sequential/conv2d/Conv2D:output:00sequential/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????882
sequential/conv2d/BiasAdd?
-sequential/batch_normalization/ReadVariableOpReadVariableOp6sequential_batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential/batch_normalization/ReadVariableOp?
/sequential/batch_normalization/ReadVariableOp_1ReadVariableOp8sequential_batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype021
/sequential/batch_normalization/ReadVariableOp_1?
>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpGsequential_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02@
>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp?
@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIsequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02B
@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?
/sequential/batch_normalization/FusedBatchNormV3FusedBatchNormV3"sequential/conv2d/BiasAdd:output:05sequential/batch_normalization/ReadVariableOp:value:07sequential/batch_normalization/ReadVariableOp_1:value:0Fsequential/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Hsequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????88:::::*
epsilon%o?:*
exponential_avg_factor%
?#<21
/sequential/batch_normalization/FusedBatchNormV3?
-sequential/batch_normalization/AssignNewValueAssignVariableOpGsequential_batch_normalization_fusedbatchnormv3_readvariableop_resource<sequential/batch_normalization/FusedBatchNormV3:batch_mean:0?^sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02/
-sequential/batch_normalization/AssignNewValue?
/sequential/batch_normalization/AssignNewValue_1AssignVariableOpIsequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource@sequential/batch_normalization/FusedBatchNormV3:batch_variance:0A^sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype021
/sequential/batch_normalization/AssignNewValue_1?
sequential/re_lu/ReluRelu3sequential/batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????882
sequential/re_lu/Relu?
)sequential/conv2d_1/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02+
)sequential/conv2d_1/Conv2D/ReadVariableOp?
sequential/conv2d_1/Conv2DConv2D#sequential/re_lu/Relu:activations:01sequential/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????88*
paddingVALID*
strides
2
sequential/conv2d_1/Conv2D?
*sequential/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*sequential/conv2d_1/BiasAdd/ReadVariableOp?
sequential/conv2d_1/BiasAddBiasAdd#sequential/conv2d_1/Conv2D:output:02sequential/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????882
sequential/conv2d_1/BiasAddr
reshape/ShapeShape$sequential/conv2d_1/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape/Shape?
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack?
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1?
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2?
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slicet
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :82
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :82
reshape/Reshape/shape/2t
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/3t
reshape/Reshape/shape/4Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/4t
reshape/Reshape/shape/5Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/5?
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0 reshape/Reshape/shape/4:output:0 reshape/Reshape/shape/5:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape?
reshape/ReshapeReshape$sequential/conv2d_1/BiasAdd:output:0reshape/Reshape/shape:output:0*
T0*7
_output_shapes%
#:!?????????882
reshape/Reshape?
ExtractImagePatchesExtractImagePatchesx*
T0*/
_output_shapes
:?????????88	*
ksizes
*
paddingSAME*
rates
*
strides
2
ExtractImagePatcheso
reshape_1/ShapeShapeExtractImagePatches:patches:0*
T0*
_output_shapes
:2
reshape_1/Shape?
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_1/strided_slice/stack?
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_1?
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_2?
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_1/strided_slicex
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :82
reshape_1/Reshape/shape/1x
reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :82
reshape_1/Reshape/shape/2x
reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/3x
reshape_1/Reshape/shape/4Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/4x
reshape_1/Reshape/shape/5Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/5?
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0"reshape_1/Reshape/shape/2:output:0"reshape_1/Reshape/shape/3:output:0"reshape_1/Reshape/shape/4:output:0"reshape_1/Reshape/shape/5:output:0*
N*
T0*
_output_shapes
:2
reshape_1/Reshape/shape?
reshape_1/ReshapeReshapeExtractImagePatches:patches:0 reshape_1/Reshape/shape:output:0*
T0*7
_output_shapes%
#:!?????????882
reshape_1/Reshape?
MulMulreshape/Reshape:output:0reshape_1/Reshape:output:0*
T0*7
_output_shapes%
#:!?????????882
Mulp
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indicesx
SumSumMul:z:0Sum/reduction_indices:output:0*
T0*3
_output_shapes!
:?????????882
Sum^
reshape_2/ShapeShapeSum:output:0*
T0*
_output_shapes
:2
reshape_2/Shape?
reshape_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_2/strided_slice/stack?
reshape_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_2/strided_slice/stack_1?
reshape_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_2/strided_slice/stack_2?
reshape_2/strided_sliceStridedSlicereshape_2/Shape:output:0&reshape_2/strided_slice/stack:output:0(reshape_2/strided_slice/stack_1:output:0(reshape_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_2/strided_slicex
reshape_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :82
reshape_2/Reshape/shape/1x
reshape_2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :82
reshape_2/Reshape/shape/2x
reshape_2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_2/Reshape/shape/3?
reshape_2/Reshape/shapePack reshape_2/strided_slice:output:0"reshape_2/Reshape/shape/1:output:0"reshape_2/Reshape/shape/2:output:0"reshape_2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_2/Reshape/shape?
reshape_2/ReshapeReshapeSum:output:0 reshape_2/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????882
reshape_2/Reshape}
IdentityIdentityreshape_2/Reshape:output:0^NoOp*
T0*/
_output_shapes
:?????????882

Identity?
NoOpNoOp.^sequential/batch_normalization/AssignNewValue0^sequential/batch_normalization/AssignNewValue_1?^sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpA^sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1.^sequential/batch_normalization/ReadVariableOp0^sequential/batch_normalization/ReadVariableOp_1)^sequential/conv2d/BiasAdd/ReadVariableOp(^sequential/conv2d/Conv2D/ReadVariableOp+^sequential/conv2d_1/BiasAdd/ReadVariableOp*^sequential/conv2d_1/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????pp: : : : : : : : 2^
-sequential/batch_normalization/AssignNewValue-sequential/batch_normalization/AssignNewValue2b
/sequential/batch_normalization/AssignNewValue_1/sequential/batch_normalization/AssignNewValue_12?
>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp2?
@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_12^
-sequential/batch_normalization/ReadVariableOp-sequential/batch_normalization/ReadVariableOp2b
/sequential/batch_normalization/ReadVariableOp_1/sequential/batch_normalization/ReadVariableOp_12T
(sequential/conv2d/BiasAdd/ReadVariableOp(sequential/conv2d/BiasAdd/ReadVariableOp2R
'sequential/conv2d/Conv2D/ReadVariableOp'sequential/conv2d/Conv2D/ReadVariableOp2X
*sequential/conv2d_1/BiasAdd/ReadVariableOp*sequential/conv2d_1/BiasAdd/ReadVariableOp2V
)sequential/conv2d_1/Conv2D/ReadVariableOp)sequential/conv2d_1/Conv2D/ReadVariableOp:R N
/
_output_shapes
:?????????pp

_user_specified_namex
?
?
F__inference_sequential_layer_call_and_return_conditional_losses_303161
conv2d_input'
conv2d_303140:
conv2d_303142:(
batch_normalization_303145:(
batch_normalization_303147:(
batch_normalization_303149:(
batch_normalization_303151:)
conv2d_1_303155:
conv2d_1_303157:
identity??+batch_normalization/StatefulPartitionedCall?conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallconv2d_inputconv2d_303140conv2d_303142*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????88*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_3029042 
conv2d/StatefulPartitionedCall?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_303145batch_normalization_303147batch_normalization_303149batch_normalization_303151*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????88*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_3029272-
+batch_normalization/StatefulPartitionedCall?
re_lu/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????88* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_re_lu_layer_call_and_return_conditional_losses_3029422
re_lu/PartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallre_lu/PartitionedCall:output:0conv2d_1_303155conv2d_1_303157*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????88*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_3029542"
 conv2d_1/StatefulPartitionedCall?
IdentityIdentity)conv2d_1/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????882

Identity?
NoOpNoOp,^batch_normalization/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????88: : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall:] Y
/
_output_shapes
:?????????88
&
_user_specified_nameconv2d_input
?
?
O__inference_batch_normalization_layer_call_and_return_conditional_losses_307553

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????88:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????882

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????88: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????88
 
_user_specified_nameinputs
?

?
+__inference_sequential_layer_call_fn_303426
conv2d_input!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5:	
	unknown_6:	
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????	**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_3034072
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????	2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:?????????
&
_user_specified_nameconv2d_input
?
D
(__inference_re_lu_1_layer_call_fn_306586

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????88* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_re_lu_1_layer_call_and_return_conditional_losses_3039202
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????882

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????88:W S
/
_output_shapes
:?????????88
 
_user_specified_nameinputs
?C
?
I__inference_InRFNet_Model_layer_call_and_return_conditional_losses_305087
input_1&
inv_1_305000:
inv_1_305002:
inv_1_305004:
inv_1_305006:
inv_1_305008:
inv_1_305010:&
inv_1_305012:	
inv_1_305014:	'
inv_22_305019:
inv_22_305021:
inv_22_305023:
inv_22_305025:
inv_22_305027:
inv_22_305029:'
inv_22_305031:
inv_22_305033:'
inv_21_305036:
inv_21_305038:
inv_21_305040:
inv_21_305042:
inv_21_305044:
inv_21_305046:'
inv_21_305048:
inv_21_305050:&
inv_3_305057:
inv_3_305059:
inv_3_305061:
inv_3_305063:
inv_3_305065:
inv_3_305067:&
inv_3_305069:	
inv_3_305071:	
dense_305076:	?$@
dense_305078:@ 
dense_1_305081:@
dense_1_305083:
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?inv_1/StatefulPartitionedCall?inv_21/StatefulPartitionedCall?inv_22/StatefulPartitionedCall?inv_3/StatefulPartitionedCall?
inv_1/StatefulPartitionedCallStatefulPartitionedCallinput_1inv_1_305000inv_1_305002inv_1_305004inv_1_305006inv_1_305008inv_1_305010inv_1_305012inv_1_305014*
Tin
2	*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_inv_1_layer_call_and_return_conditional_losses_3037082
inv_1/StatefulPartitionedCall?
re_lu/PartitionedCallPartitionedCall&inv_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_re_lu_layer_call_and_return_conditional_losses_3037312
re_lu/PartitionedCall?
max_pooling2d/PartitionedCallPartitionedCallre_lu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????pp* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_3037372
max_pooling2d/PartitionedCall?
inv_22/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0inv_22_305019inv_22_305021inv_22_305023inv_22_305025inv_22_305027inv_22_305029inv_22_305031inv_22_305033*
Tin
2	*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????88**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_inv_22_layer_call_and_return_conditional_losses_3038092 
inv_22/StatefulPartitionedCall?
inv_21/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0inv_21_305036inv_21_305038inv_21_305040inv_21_305042inv_21_305044inv_21_305046inv_21_305048inv_21_305050*
Tin
2	*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????88**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_inv_21_layer_call_and_return_conditional_losses_3038972 
inv_21/StatefulPartitionedCall?
re_lu_1/PartitionedCallPartitionedCall'inv_21/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????88* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_re_lu_1_layer_call_and_return_conditional_losses_3039202
re_lu_1/PartitionedCall?
re_lu_2/PartitionedCallPartitionedCall'inv_22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????88* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_re_lu_2_layer_call_and_return_conditional_losses_3039272
re_lu_2/PartitionedCall?
concatenate/PartitionedCallPartitionedCall re_lu_1/PartitionedCall:output:0 re_lu_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????88* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_3039362
concatenate/PartitionedCall?
max_pooling2d_1/PartitionedCallPartitionedCall$concatenate/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_3039422!
max_pooling2d_1/PartitionedCall?
inv_3/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0inv_3_305057inv_3_305059inv_3_305061inv_3_305063inv_3_305065inv_3_305067inv_3_305069inv_3_305071*
Tin
2	*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_inv_3_layer_call_and_return_conditional_losses_3040142
inv_3/StatefulPartitionedCall?
re_lu_3/PartitionedCallPartitionedCall&inv_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_re_lu_3_layer_call_and_return_conditional_losses_3040372
re_lu_3/PartitionedCall?
flatten/PartitionedCallPartitionedCall re_lu_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????$* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_3040452
flatten/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_305076dense_305078*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_3040582
dense/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_305081dense_1_305083*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_3040752!
dense_1/StatefulPartitionedCall?
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall^inv_1/StatefulPartitionedCall^inv_21/StatefulPartitionedCall^inv_22/StatefulPartitionedCall^inv_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*x
_input_shapesg
e:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2>
inv_1/StatefulPartitionedCallinv_1/StatefulPartitionedCall2@
inv_21/StatefulPartitionedCallinv_21/StatefulPartitionedCall2@
inv_22/StatefulPartitionedCallinv_22/StatefulPartitionedCall2>
inv_3/StatefulPartitionedCallinv_3/StatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
?

?
'__inference_inv_21_layer_call_fn_306373
x!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????88**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_inv_21_layer_call_and_return_conditional_losses_3038972
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????882

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????pp: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
/
_output_shapes
:?????????pp

_user_specified_namex
?
?
.__inference_InRFNet_Model_layer_call_fn_304157
input_1!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5:	
	unknown_6:	#
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:$

unknown_13:

unknown_14:$

unknown_15:

unknown_16:

unknown_17:

unknown_18:

unknown_19:

unknown_20:$

unknown_21:

unknown_22:$

unknown_23:

unknown_24:

unknown_25:

unknown_26:

unknown_27:

unknown_28:$

unknown_29:	

unknown_30:	

unknown_31:	?$@

unknown_32:@

unknown_33:@

unknown_34:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*F
_read_only_resource_inputs(
&$	
 !"#$*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_InRFNet_Model_layer_call_and_return_conditional_losses_3040822
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*x
_input_shapesg
e:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
??
?&
I__inference_InRFNet_Model_layer_call_and_return_conditional_losses_305554

inputsP
6inv_1_sequential_conv2d_conv2d_readvariableop_resource:E
7inv_1_sequential_conv2d_biasadd_readvariableop_resource:J
<inv_1_sequential_batch_normalization_readvariableop_resource:L
>inv_1_sequential_batch_normalization_readvariableop_1_resource:[
Minv_1_sequential_batch_normalization_fusedbatchnormv3_readvariableop_resource:]
Oinv_1_sequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource:R
8inv_1_sequential_conv2d_1_conv2d_readvariableop_resource:	G
9inv_1_sequential_conv2d_1_biasadd_readvariableop_resource:	Q
7inv_22_sequential_conv2d_conv2d_readvariableop_resource:F
8inv_22_sequential_conv2d_biasadd_readvariableop_resource:K
=inv_22_sequential_batch_normalization_readvariableop_resource:M
?inv_22_sequential_batch_normalization_readvariableop_1_resource:\
Ninv_22_sequential_batch_normalization_fusedbatchnormv3_readvariableop_resource:^
Pinv_22_sequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource:S
9inv_22_sequential_conv2d_1_conv2d_readvariableop_resource:H
:inv_22_sequential_conv2d_1_biasadd_readvariableop_resource:Q
7inv_21_sequential_conv2d_conv2d_readvariableop_resource:F
8inv_21_sequential_conv2d_biasadd_readvariableop_resource:K
=inv_21_sequential_batch_normalization_readvariableop_resource:M
?inv_21_sequential_batch_normalization_readvariableop_1_resource:\
Ninv_21_sequential_batch_normalization_fusedbatchnormv3_readvariableop_resource:^
Pinv_21_sequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource:S
9inv_21_sequential_conv2d_1_conv2d_readvariableop_resource:H
:inv_21_sequential_conv2d_1_biasadd_readvariableop_resource:P
6inv_3_sequential_conv2d_conv2d_readvariableop_resource:E
7inv_3_sequential_conv2d_biasadd_readvariableop_resource:J
<inv_3_sequential_batch_normalization_readvariableop_resource:L
>inv_3_sequential_batch_normalization_readvariableop_1_resource:[
Minv_3_sequential_batch_normalization_fusedbatchnormv3_readvariableop_resource:]
Oinv_3_sequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource:R
8inv_3_sequential_conv2d_1_conv2d_readvariableop_resource:	G
9inv_3_sequential_conv2d_1_biasadd_readvariableop_resource:	7
$dense_matmul_readvariableop_resource:	?$@3
%dense_biasadd_readvariableop_resource:@8
&dense_1_matmul_readvariableop_resource:@5
'dense_1_biasadd_readvariableop_resource:
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?Dinv_1/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp?Finv_1/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?3inv_1/sequential/batch_normalization/ReadVariableOp?5inv_1/sequential/batch_normalization/ReadVariableOp_1?.inv_1/sequential/conv2d/BiasAdd/ReadVariableOp?-inv_1/sequential/conv2d/Conv2D/ReadVariableOp?0inv_1/sequential/conv2d_1/BiasAdd/ReadVariableOp?/inv_1/sequential/conv2d_1/Conv2D/ReadVariableOp?Einv_21/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp?Ginv_21/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?4inv_21/sequential/batch_normalization/ReadVariableOp?6inv_21/sequential/batch_normalization/ReadVariableOp_1?/inv_21/sequential/conv2d/BiasAdd/ReadVariableOp?.inv_21/sequential/conv2d/Conv2D/ReadVariableOp?1inv_21/sequential/conv2d_1/BiasAdd/ReadVariableOp?0inv_21/sequential/conv2d_1/Conv2D/ReadVariableOp?Einv_22/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp?Ginv_22/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?4inv_22/sequential/batch_normalization/ReadVariableOp?6inv_22/sequential/batch_normalization/ReadVariableOp_1?/inv_22/sequential/conv2d/BiasAdd/ReadVariableOp?.inv_22/sequential/conv2d/Conv2D/ReadVariableOp?1inv_22/sequential/conv2d_1/BiasAdd/ReadVariableOp?0inv_22/sequential/conv2d_1/Conv2D/ReadVariableOp?Dinv_3/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp?Finv_3/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?3inv_3/sequential/batch_normalization/ReadVariableOp?5inv_3/sequential/batch_normalization/ReadVariableOp_1?.inv_3/sequential/conv2d/BiasAdd/ReadVariableOp?-inv_3/sequential/conv2d/Conv2D/ReadVariableOp?0inv_3/sequential/conv2d_1/BiasAdd/ReadVariableOp?/inv_3/sequential/conv2d_1/Conv2D/ReadVariableOpp
inv_1/IdentityIdentityinputs*
T0*1
_output_shapes
:???????????2
inv_1/Identity?
-inv_1/sequential/conv2d/Conv2D/ReadVariableOpReadVariableOp6inv_1_sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02/
-inv_1/sequential/conv2d/Conv2D/ReadVariableOp?
inv_1/sequential/conv2d/Conv2DConv2Dinv_1/Identity:output:05inv_1/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2 
inv_1/sequential/conv2d/Conv2D?
.inv_1/sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOp7inv_1_sequential_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.inv_1/sequential/conv2d/BiasAdd/ReadVariableOp?
inv_1/sequential/conv2d/BiasAddBiasAdd'inv_1/sequential/conv2d/Conv2D:output:06inv_1/sequential/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2!
inv_1/sequential/conv2d/BiasAdd?
3inv_1/sequential/batch_normalization/ReadVariableOpReadVariableOp<inv_1_sequential_batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype025
3inv_1/sequential/batch_normalization/ReadVariableOp?
5inv_1/sequential/batch_normalization/ReadVariableOp_1ReadVariableOp>inv_1_sequential_batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype027
5inv_1/sequential/batch_normalization/ReadVariableOp_1?
Dinv_1/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpMinv_1_sequential_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02F
Dinv_1/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp?
Finv_1/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOinv_1_sequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02H
Finv_1/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?
5inv_1/sequential/batch_normalization/FusedBatchNormV3FusedBatchNormV3(inv_1/sequential/conv2d/BiasAdd:output:0;inv_1/sequential/batch_normalization/ReadVariableOp:value:0=inv_1/sequential/batch_normalization/ReadVariableOp_1:value:0Linv_1/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Ninv_1/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( 27
5inv_1/sequential/batch_normalization/FusedBatchNormV3?
inv_1/sequential/re_lu/ReluRelu9inv_1/sequential/batch_normalization/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????2
inv_1/sequential/re_lu/Relu?
/inv_1/sequential/conv2d_1/Conv2D/ReadVariableOpReadVariableOp8inv_1_sequential_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:	*
dtype021
/inv_1/sequential/conv2d_1/Conv2D/ReadVariableOp?
 inv_1/sequential/conv2d_1/Conv2DConv2D)inv_1/sequential/re_lu/Relu:activations:07inv_1/sequential/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????	*
paddingVALID*
strides
2"
 inv_1/sequential/conv2d_1/Conv2D?
0inv_1/sequential/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp9inv_1_sequential_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype022
0inv_1/sequential/conv2d_1/BiasAdd/ReadVariableOp?
!inv_1/sequential/conv2d_1/BiasAddBiasAdd)inv_1/sequential/conv2d_1/Conv2D:output:08inv_1/sequential/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????	2#
!inv_1/sequential/conv2d_1/BiasAdd?
inv_1/reshape/ShapeShape*inv_1/sequential/conv2d_1/BiasAdd:output:0*
T0*
_output_shapes
:2
inv_1/reshape/Shape?
!inv_1/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2#
!inv_1/reshape/strided_slice/stack?
#inv_1/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#inv_1/reshape/strided_slice/stack_1?
#inv_1/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#inv_1/reshape/strided_slice/stack_2?
inv_1/reshape/strided_sliceStridedSliceinv_1/reshape/Shape:output:0*inv_1/reshape/strided_slice/stack:output:0,inv_1/reshape/strided_slice/stack_1:output:0,inv_1/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
inv_1/reshape/strided_slice?
inv_1/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :?2
inv_1/reshape/Reshape/shape/1?
inv_1/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :?2
inv_1/reshape/Reshape/shape/2?
inv_1/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :	2
inv_1/reshape/Reshape/shape/3?
inv_1/reshape/Reshape/shape/4Const*
_output_shapes
: *
dtype0*
value	B :2
inv_1/reshape/Reshape/shape/4?
inv_1/reshape/Reshape/shape/5Const*
_output_shapes
: *
dtype0*
value	B :2
inv_1/reshape/Reshape/shape/5?
inv_1/reshape/Reshape/shapePack$inv_1/reshape/strided_slice:output:0&inv_1/reshape/Reshape/shape/1:output:0&inv_1/reshape/Reshape/shape/2:output:0&inv_1/reshape/Reshape/shape/3:output:0&inv_1/reshape/Reshape/shape/4:output:0&inv_1/reshape/Reshape/shape/5:output:0*
N*
T0*
_output_shapes
:2
inv_1/reshape/Reshape/shape?
inv_1/reshape/ReshapeReshape*inv_1/sequential/conv2d_1/BiasAdd:output:0$inv_1/reshape/Reshape/shape:output:0*
T0*9
_output_shapes'
%:#???????????	2
inv_1/reshape/Reshape?
inv_1/ExtractImagePatchesExtractImagePatchesinputs*
T0*1
_output_shapes
:???????????*
ksizes
*
paddingSAME*
rates
*
strides
2
inv_1/ExtractImagePatches?
inv_1/reshape_1/ShapeShape#inv_1/ExtractImagePatches:patches:0*
T0*
_output_shapes
:2
inv_1/reshape_1/Shape?
#inv_1/reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#inv_1/reshape_1/strided_slice/stack?
%inv_1/reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%inv_1/reshape_1/strided_slice/stack_1?
%inv_1/reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%inv_1/reshape_1/strided_slice/stack_2?
inv_1/reshape_1/strided_sliceStridedSliceinv_1/reshape_1/Shape:output:0,inv_1/reshape_1/strided_slice/stack:output:0.inv_1/reshape_1/strided_slice/stack_1:output:0.inv_1/reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
inv_1/reshape_1/strided_slice?
inv_1/reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :?2!
inv_1/reshape_1/Reshape/shape/1?
inv_1/reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :?2!
inv_1/reshape_1/Reshape/shape/2?
inv_1/reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :	2!
inv_1/reshape_1/Reshape/shape/3?
inv_1/reshape_1/Reshape/shape/4Const*
_output_shapes
: *
dtype0*
value	B :2!
inv_1/reshape_1/Reshape/shape/4?
inv_1/reshape_1/Reshape/shape/5Const*
_output_shapes
: *
dtype0*
value	B :2!
inv_1/reshape_1/Reshape/shape/5?
inv_1/reshape_1/Reshape/shapePack&inv_1/reshape_1/strided_slice:output:0(inv_1/reshape_1/Reshape/shape/1:output:0(inv_1/reshape_1/Reshape/shape/2:output:0(inv_1/reshape_1/Reshape/shape/3:output:0(inv_1/reshape_1/Reshape/shape/4:output:0(inv_1/reshape_1/Reshape/shape/5:output:0*
N*
T0*
_output_shapes
:2
inv_1/reshape_1/Reshape/shape?
inv_1/reshape_1/ReshapeReshape#inv_1/ExtractImagePatches:patches:0&inv_1/reshape_1/Reshape/shape:output:0*
T0*9
_output_shapes'
%:#???????????	2
inv_1/reshape_1/Reshape?
	inv_1/MulMulinv_1/reshape/Reshape:output:0 inv_1/reshape_1/Reshape:output:0*
T0*9
_output_shapes'
%:#???????????	2
	inv_1/Mul|
inv_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
inv_1/Sum/reduction_indices?
	inv_1/SumSuminv_1/Mul:z:0$inv_1/Sum/reduction_indices:output:0*
T0*5
_output_shapes#
!:???????????2
	inv_1/Sump
inv_1/reshape_2/ShapeShapeinv_1/Sum:output:0*
T0*
_output_shapes
:2
inv_1/reshape_2/Shape?
#inv_1/reshape_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#inv_1/reshape_2/strided_slice/stack?
%inv_1/reshape_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%inv_1/reshape_2/strided_slice/stack_1?
%inv_1/reshape_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%inv_1/reshape_2/strided_slice/stack_2?
inv_1/reshape_2/strided_sliceStridedSliceinv_1/reshape_2/Shape:output:0,inv_1/reshape_2/strided_slice/stack:output:0.inv_1/reshape_2/strided_slice/stack_1:output:0.inv_1/reshape_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
inv_1/reshape_2/strided_slice?
inv_1/reshape_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :?2!
inv_1/reshape_2/Reshape/shape/1?
inv_1/reshape_2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :?2!
inv_1/reshape_2/Reshape/shape/2?
inv_1/reshape_2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2!
inv_1/reshape_2/Reshape/shape/3?
inv_1/reshape_2/Reshape/shapePack&inv_1/reshape_2/strided_slice:output:0(inv_1/reshape_2/Reshape/shape/1:output:0(inv_1/reshape_2/Reshape/shape/2:output:0(inv_1/reshape_2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
inv_1/reshape_2/Reshape/shape?
inv_1/reshape_2/ReshapeReshapeinv_1/Sum:output:0&inv_1/reshape_2/Reshape/shape:output:0*
T0*1
_output_shapes
:???????????2
inv_1/reshape_2/Reshape~

re_lu/ReluRelu inv_1/reshape_2/Reshape:output:0*
T0*1
_output_shapes
:???????????2

re_lu/Relu?
max_pooling2d/MaxPoolMaxPoolre_lu/Relu:activations:0*/
_output_shapes
:?????????pp*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPool?
 inv_22/average_pooling2d/AvgPoolAvgPoolmax_pooling2d/MaxPool:output:0*
T0*/
_output_shapes
:?????????88*
ksize
*
paddingSAME*
strides
2"
 inv_22/average_pooling2d/AvgPool?
.inv_22/sequential/conv2d/Conv2D/ReadVariableOpReadVariableOp7inv_22_sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype020
.inv_22/sequential/conv2d/Conv2D/ReadVariableOp?
inv_22/sequential/conv2d/Conv2DConv2D)inv_22/average_pooling2d/AvgPool:output:06inv_22/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????88*
paddingVALID*
strides
2!
inv_22/sequential/conv2d/Conv2D?
/inv_22/sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOp8inv_22_sequential_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/inv_22/sequential/conv2d/BiasAdd/ReadVariableOp?
 inv_22/sequential/conv2d/BiasAddBiasAdd(inv_22/sequential/conv2d/Conv2D:output:07inv_22/sequential/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????882"
 inv_22/sequential/conv2d/BiasAdd?
4inv_22/sequential/batch_normalization/ReadVariableOpReadVariableOp=inv_22_sequential_batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype026
4inv_22/sequential/batch_normalization/ReadVariableOp?
6inv_22/sequential/batch_normalization/ReadVariableOp_1ReadVariableOp?inv_22_sequential_batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype028
6inv_22/sequential/batch_normalization/ReadVariableOp_1?
Einv_22/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpNinv_22_sequential_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02G
Einv_22/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp?
Ginv_22/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpPinv_22_sequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02I
Ginv_22/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?
6inv_22/sequential/batch_normalization/FusedBatchNormV3FusedBatchNormV3)inv_22/sequential/conv2d/BiasAdd:output:0<inv_22/sequential/batch_normalization/ReadVariableOp:value:0>inv_22/sequential/batch_normalization/ReadVariableOp_1:value:0Minv_22/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Oinv_22/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????88:::::*
epsilon%o?:*
is_training( 28
6inv_22/sequential/batch_normalization/FusedBatchNormV3?
inv_22/sequential/re_lu/ReluRelu:inv_22/sequential/batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????882
inv_22/sequential/re_lu/Relu?
0inv_22/sequential/conv2d_1/Conv2D/ReadVariableOpReadVariableOp9inv_22_sequential_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype022
0inv_22/sequential/conv2d_1/Conv2D/ReadVariableOp?
!inv_22/sequential/conv2d_1/Conv2DConv2D*inv_22/sequential/re_lu/Relu:activations:08inv_22/sequential/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????88*
paddingVALID*
strides
2#
!inv_22/sequential/conv2d_1/Conv2D?
1inv_22/sequential/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp:inv_22_sequential_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype023
1inv_22/sequential/conv2d_1/BiasAdd/ReadVariableOp?
"inv_22/sequential/conv2d_1/BiasAddBiasAdd*inv_22/sequential/conv2d_1/Conv2D:output:09inv_22/sequential/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????882$
"inv_22/sequential/conv2d_1/BiasAdd?
inv_22/reshape/ShapeShape+inv_22/sequential/conv2d_1/BiasAdd:output:0*
T0*
_output_shapes
:2
inv_22/reshape/Shape?
"inv_22/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"inv_22/reshape/strided_slice/stack?
$inv_22/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$inv_22/reshape/strided_slice/stack_1?
$inv_22/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$inv_22/reshape/strided_slice/stack_2?
inv_22/reshape/strided_sliceStridedSliceinv_22/reshape/Shape:output:0+inv_22/reshape/strided_slice/stack:output:0-inv_22/reshape/strided_slice/stack_1:output:0-inv_22/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
inv_22/reshape/strided_slice?
inv_22/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :82 
inv_22/reshape/Reshape/shape/1?
inv_22/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :82 
inv_22/reshape/Reshape/shape/2?
inv_22/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2 
inv_22/reshape/Reshape/shape/3?
inv_22/reshape/Reshape/shape/4Const*
_output_shapes
: *
dtype0*
value	B :2 
inv_22/reshape/Reshape/shape/4?
inv_22/reshape/Reshape/shape/5Const*
_output_shapes
: *
dtype0*
value	B :2 
inv_22/reshape/Reshape/shape/5?
inv_22/reshape/Reshape/shapePack%inv_22/reshape/strided_slice:output:0'inv_22/reshape/Reshape/shape/1:output:0'inv_22/reshape/Reshape/shape/2:output:0'inv_22/reshape/Reshape/shape/3:output:0'inv_22/reshape/Reshape/shape/4:output:0'inv_22/reshape/Reshape/shape/5:output:0*
N*
T0*
_output_shapes
:2
inv_22/reshape/Reshape/shape?
inv_22/reshape/ReshapeReshape+inv_22/sequential/conv2d_1/BiasAdd:output:0%inv_22/reshape/Reshape/shape:output:0*
T0*7
_output_shapes%
#:!?????????882
inv_22/reshape/Reshape?
inv_22/ExtractImagePatchesExtractImagePatchesmax_pooling2d/MaxPool:output:0*
T0*/
_output_shapes
:?????????88	*
ksizes
*
paddingSAME*
rates
*
strides
2
inv_22/ExtractImagePatches?
inv_22/reshape_1/ShapeShape$inv_22/ExtractImagePatches:patches:0*
T0*
_output_shapes
:2
inv_22/reshape_1/Shape?
$inv_22/reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$inv_22/reshape_1/strided_slice/stack?
&inv_22/reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&inv_22/reshape_1/strided_slice/stack_1?
&inv_22/reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&inv_22/reshape_1/strided_slice/stack_2?
inv_22/reshape_1/strided_sliceStridedSliceinv_22/reshape_1/Shape:output:0-inv_22/reshape_1/strided_slice/stack:output:0/inv_22/reshape_1/strided_slice/stack_1:output:0/inv_22/reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
inv_22/reshape_1/strided_slice?
 inv_22/reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :82"
 inv_22/reshape_1/Reshape/shape/1?
 inv_22/reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :82"
 inv_22/reshape_1/Reshape/shape/2?
 inv_22/reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2"
 inv_22/reshape_1/Reshape/shape/3?
 inv_22/reshape_1/Reshape/shape/4Const*
_output_shapes
: *
dtype0*
value	B :2"
 inv_22/reshape_1/Reshape/shape/4?
 inv_22/reshape_1/Reshape/shape/5Const*
_output_shapes
: *
dtype0*
value	B :2"
 inv_22/reshape_1/Reshape/shape/5?
inv_22/reshape_1/Reshape/shapePack'inv_22/reshape_1/strided_slice:output:0)inv_22/reshape_1/Reshape/shape/1:output:0)inv_22/reshape_1/Reshape/shape/2:output:0)inv_22/reshape_1/Reshape/shape/3:output:0)inv_22/reshape_1/Reshape/shape/4:output:0)inv_22/reshape_1/Reshape/shape/5:output:0*
N*
T0*
_output_shapes
:2 
inv_22/reshape_1/Reshape/shape?
inv_22/reshape_1/ReshapeReshape$inv_22/ExtractImagePatches:patches:0'inv_22/reshape_1/Reshape/shape:output:0*
T0*7
_output_shapes%
#:!?????????882
inv_22/reshape_1/Reshape?

inv_22/MulMulinv_22/reshape/Reshape:output:0!inv_22/reshape_1/Reshape:output:0*
T0*7
_output_shapes%
#:!?????????882

inv_22/Mul~
inv_22/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
inv_22/Sum/reduction_indices?

inv_22/SumSuminv_22/Mul:z:0%inv_22/Sum/reduction_indices:output:0*
T0*3
_output_shapes!
:?????????882

inv_22/Sums
inv_22/reshape_2/ShapeShapeinv_22/Sum:output:0*
T0*
_output_shapes
:2
inv_22/reshape_2/Shape?
$inv_22/reshape_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$inv_22/reshape_2/strided_slice/stack?
&inv_22/reshape_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&inv_22/reshape_2/strided_slice/stack_1?
&inv_22/reshape_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&inv_22/reshape_2/strided_slice/stack_2?
inv_22/reshape_2/strided_sliceStridedSliceinv_22/reshape_2/Shape:output:0-inv_22/reshape_2/strided_slice/stack:output:0/inv_22/reshape_2/strided_slice/stack_1:output:0/inv_22/reshape_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
inv_22/reshape_2/strided_slice?
 inv_22/reshape_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :82"
 inv_22/reshape_2/Reshape/shape/1?
 inv_22/reshape_2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :82"
 inv_22/reshape_2/Reshape/shape/2?
 inv_22/reshape_2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2"
 inv_22/reshape_2/Reshape/shape/3?
inv_22/reshape_2/Reshape/shapePack'inv_22/reshape_2/strided_slice:output:0)inv_22/reshape_2/Reshape/shape/1:output:0)inv_22/reshape_2/Reshape/shape/2:output:0)inv_22/reshape_2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2 
inv_22/reshape_2/Reshape/shape?
inv_22/reshape_2/ReshapeReshapeinv_22/Sum:output:0'inv_22/reshape_2/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????882
inv_22/reshape_2/Reshape?
 inv_21/average_pooling2d/AvgPoolAvgPoolmax_pooling2d/MaxPool:output:0*
T0*/
_output_shapes
:?????????88*
ksize
*
paddingSAME*
strides
2"
 inv_21/average_pooling2d/AvgPool?
.inv_21/sequential/conv2d/Conv2D/ReadVariableOpReadVariableOp7inv_21_sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype020
.inv_21/sequential/conv2d/Conv2D/ReadVariableOp?
inv_21/sequential/conv2d/Conv2DConv2D)inv_21/average_pooling2d/AvgPool:output:06inv_21/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????88*
paddingVALID*
strides
2!
inv_21/sequential/conv2d/Conv2D?
/inv_21/sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOp8inv_21_sequential_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/inv_21/sequential/conv2d/BiasAdd/ReadVariableOp?
 inv_21/sequential/conv2d/BiasAddBiasAdd(inv_21/sequential/conv2d/Conv2D:output:07inv_21/sequential/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????882"
 inv_21/sequential/conv2d/BiasAdd?
4inv_21/sequential/batch_normalization/ReadVariableOpReadVariableOp=inv_21_sequential_batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype026
4inv_21/sequential/batch_normalization/ReadVariableOp?
6inv_21/sequential/batch_normalization/ReadVariableOp_1ReadVariableOp?inv_21_sequential_batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype028
6inv_21/sequential/batch_normalization/ReadVariableOp_1?
Einv_21/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpNinv_21_sequential_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02G
Einv_21/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp?
Ginv_21/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpPinv_21_sequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02I
Ginv_21/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?
6inv_21/sequential/batch_normalization/FusedBatchNormV3FusedBatchNormV3)inv_21/sequential/conv2d/BiasAdd:output:0<inv_21/sequential/batch_normalization/ReadVariableOp:value:0>inv_21/sequential/batch_normalization/ReadVariableOp_1:value:0Minv_21/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Oinv_21/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????88:::::*
epsilon%o?:*
is_training( 28
6inv_21/sequential/batch_normalization/FusedBatchNormV3?
inv_21/sequential/re_lu/ReluRelu:inv_21/sequential/batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????882
inv_21/sequential/re_lu/Relu?
0inv_21/sequential/conv2d_1/Conv2D/ReadVariableOpReadVariableOp9inv_21_sequential_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype022
0inv_21/sequential/conv2d_1/Conv2D/ReadVariableOp?
!inv_21/sequential/conv2d_1/Conv2DConv2D*inv_21/sequential/re_lu/Relu:activations:08inv_21/sequential/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????88*
paddingVALID*
strides
2#
!inv_21/sequential/conv2d_1/Conv2D?
1inv_21/sequential/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp:inv_21_sequential_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype023
1inv_21/sequential/conv2d_1/BiasAdd/ReadVariableOp?
"inv_21/sequential/conv2d_1/BiasAddBiasAdd*inv_21/sequential/conv2d_1/Conv2D:output:09inv_21/sequential/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????882$
"inv_21/sequential/conv2d_1/BiasAdd?
inv_21/reshape/ShapeShape+inv_21/sequential/conv2d_1/BiasAdd:output:0*
T0*
_output_shapes
:2
inv_21/reshape/Shape?
"inv_21/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"inv_21/reshape/strided_slice/stack?
$inv_21/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$inv_21/reshape/strided_slice/stack_1?
$inv_21/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$inv_21/reshape/strided_slice/stack_2?
inv_21/reshape/strided_sliceStridedSliceinv_21/reshape/Shape:output:0+inv_21/reshape/strided_slice/stack:output:0-inv_21/reshape/strided_slice/stack_1:output:0-inv_21/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
inv_21/reshape/strided_slice?
inv_21/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :82 
inv_21/reshape/Reshape/shape/1?
inv_21/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :82 
inv_21/reshape/Reshape/shape/2?
inv_21/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2 
inv_21/reshape/Reshape/shape/3?
inv_21/reshape/Reshape/shape/4Const*
_output_shapes
: *
dtype0*
value	B :2 
inv_21/reshape/Reshape/shape/4?
inv_21/reshape/Reshape/shape/5Const*
_output_shapes
: *
dtype0*
value	B :2 
inv_21/reshape/Reshape/shape/5?
inv_21/reshape/Reshape/shapePack%inv_21/reshape/strided_slice:output:0'inv_21/reshape/Reshape/shape/1:output:0'inv_21/reshape/Reshape/shape/2:output:0'inv_21/reshape/Reshape/shape/3:output:0'inv_21/reshape/Reshape/shape/4:output:0'inv_21/reshape/Reshape/shape/5:output:0*
N*
T0*
_output_shapes
:2
inv_21/reshape/Reshape/shape?
inv_21/reshape/ReshapeReshape+inv_21/sequential/conv2d_1/BiasAdd:output:0%inv_21/reshape/Reshape/shape:output:0*
T0*7
_output_shapes%
#:!?????????882
inv_21/reshape/Reshape?
inv_21/ExtractImagePatchesExtractImagePatchesmax_pooling2d/MaxPool:output:0*
T0*/
_output_shapes
:?????????88	*
ksizes
*
paddingSAME*
rates
*
strides
2
inv_21/ExtractImagePatches?
inv_21/reshape_1/ShapeShape$inv_21/ExtractImagePatches:patches:0*
T0*
_output_shapes
:2
inv_21/reshape_1/Shape?
$inv_21/reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$inv_21/reshape_1/strided_slice/stack?
&inv_21/reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&inv_21/reshape_1/strided_slice/stack_1?
&inv_21/reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&inv_21/reshape_1/strided_slice/stack_2?
inv_21/reshape_1/strided_sliceStridedSliceinv_21/reshape_1/Shape:output:0-inv_21/reshape_1/strided_slice/stack:output:0/inv_21/reshape_1/strided_slice/stack_1:output:0/inv_21/reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
inv_21/reshape_1/strided_slice?
 inv_21/reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :82"
 inv_21/reshape_1/Reshape/shape/1?
 inv_21/reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :82"
 inv_21/reshape_1/Reshape/shape/2?
 inv_21/reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2"
 inv_21/reshape_1/Reshape/shape/3?
 inv_21/reshape_1/Reshape/shape/4Const*
_output_shapes
: *
dtype0*
value	B :2"
 inv_21/reshape_1/Reshape/shape/4?
 inv_21/reshape_1/Reshape/shape/5Const*
_output_shapes
: *
dtype0*
value	B :2"
 inv_21/reshape_1/Reshape/shape/5?
inv_21/reshape_1/Reshape/shapePack'inv_21/reshape_1/strided_slice:output:0)inv_21/reshape_1/Reshape/shape/1:output:0)inv_21/reshape_1/Reshape/shape/2:output:0)inv_21/reshape_1/Reshape/shape/3:output:0)inv_21/reshape_1/Reshape/shape/4:output:0)inv_21/reshape_1/Reshape/shape/5:output:0*
N*
T0*
_output_shapes
:2 
inv_21/reshape_1/Reshape/shape?
inv_21/reshape_1/ReshapeReshape$inv_21/ExtractImagePatches:patches:0'inv_21/reshape_1/Reshape/shape:output:0*
T0*7
_output_shapes%
#:!?????????882
inv_21/reshape_1/Reshape?

inv_21/MulMulinv_21/reshape/Reshape:output:0!inv_21/reshape_1/Reshape:output:0*
T0*7
_output_shapes%
#:!?????????882

inv_21/Mul~
inv_21/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
inv_21/Sum/reduction_indices?

inv_21/SumSuminv_21/Mul:z:0%inv_21/Sum/reduction_indices:output:0*
T0*3
_output_shapes!
:?????????882

inv_21/Sums
inv_21/reshape_2/ShapeShapeinv_21/Sum:output:0*
T0*
_output_shapes
:2
inv_21/reshape_2/Shape?
$inv_21/reshape_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$inv_21/reshape_2/strided_slice/stack?
&inv_21/reshape_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&inv_21/reshape_2/strided_slice/stack_1?
&inv_21/reshape_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&inv_21/reshape_2/strided_slice/stack_2?
inv_21/reshape_2/strided_sliceStridedSliceinv_21/reshape_2/Shape:output:0-inv_21/reshape_2/strided_slice/stack:output:0/inv_21/reshape_2/strided_slice/stack_1:output:0/inv_21/reshape_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
inv_21/reshape_2/strided_slice?
 inv_21/reshape_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :82"
 inv_21/reshape_2/Reshape/shape/1?
 inv_21/reshape_2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :82"
 inv_21/reshape_2/Reshape/shape/2?
 inv_21/reshape_2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2"
 inv_21/reshape_2/Reshape/shape/3?
inv_21/reshape_2/Reshape/shapePack'inv_21/reshape_2/strided_slice:output:0)inv_21/reshape_2/Reshape/shape/1:output:0)inv_21/reshape_2/Reshape/shape/2:output:0)inv_21/reshape_2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2 
inv_21/reshape_2/Reshape/shape?
inv_21/reshape_2/ReshapeReshapeinv_21/Sum:output:0'inv_21/reshape_2/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????882
inv_21/reshape_2/Reshape?
re_lu_1/ReluRelu!inv_21/reshape_2/Reshape:output:0*
T0*/
_output_shapes
:?????????882
re_lu_1/Relu?
re_lu_2/ReluRelu!inv_22/reshape_2/Reshape:output:0*
T0*/
_output_shapes
:?????????882
re_lu_2/Relut
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis?
concatenate/concatConcatV2re_lu_1/Relu:activations:0re_lu_2/Relu:activations:0 concatenate/concat/axis:output:0*
N*
T0*/
_output_shapes
:?????????882
concatenate/concat?
max_pooling2d_1/MaxPoolMaxPoolconcatenate/concat:output:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPool?
inv_3/IdentityIdentity max_pooling2d_1/MaxPool:output:0*
T0*/
_output_shapes
:?????????2
inv_3/Identity?
-inv_3/sequential/conv2d/Conv2D/ReadVariableOpReadVariableOp6inv_3_sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02/
-inv_3/sequential/conv2d/Conv2D/ReadVariableOp?
inv_3/sequential/conv2d/Conv2DConv2Dinv_3/Identity:output:05inv_3/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2 
inv_3/sequential/conv2d/Conv2D?
.inv_3/sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOp7inv_3_sequential_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.inv_3/sequential/conv2d/BiasAdd/ReadVariableOp?
inv_3/sequential/conv2d/BiasAddBiasAdd'inv_3/sequential/conv2d/Conv2D:output:06inv_3/sequential/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2!
inv_3/sequential/conv2d/BiasAdd?
3inv_3/sequential/batch_normalization/ReadVariableOpReadVariableOp<inv_3_sequential_batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype025
3inv_3/sequential/batch_normalization/ReadVariableOp?
5inv_3/sequential/batch_normalization/ReadVariableOp_1ReadVariableOp>inv_3_sequential_batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype027
5inv_3/sequential/batch_normalization/ReadVariableOp_1?
Dinv_3/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpMinv_3_sequential_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02F
Dinv_3/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp?
Finv_3/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOinv_3_sequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02H
Finv_3/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?
5inv_3/sequential/batch_normalization/FusedBatchNormV3FusedBatchNormV3(inv_3/sequential/conv2d/BiasAdd:output:0;inv_3/sequential/batch_normalization/ReadVariableOp:value:0=inv_3/sequential/batch_normalization/ReadVariableOp_1:value:0Linv_3/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Ninv_3/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
is_training( 27
5inv_3/sequential/batch_normalization/FusedBatchNormV3?
inv_3/sequential/re_lu/ReluRelu9inv_3/sequential/batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????2
inv_3/sequential/re_lu/Relu?
/inv_3/sequential/conv2d_1/Conv2D/ReadVariableOpReadVariableOp8inv_3_sequential_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:	*
dtype021
/inv_3/sequential/conv2d_1/Conv2D/ReadVariableOp?
 inv_3/sequential/conv2d_1/Conv2DConv2D)inv_3/sequential/re_lu/Relu:activations:07inv_3/sequential/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????	*
paddingVALID*
strides
2"
 inv_3/sequential/conv2d_1/Conv2D?
0inv_3/sequential/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp9inv_3_sequential_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype022
0inv_3/sequential/conv2d_1/BiasAdd/ReadVariableOp?
!inv_3/sequential/conv2d_1/BiasAddBiasAdd)inv_3/sequential/conv2d_1/Conv2D:output:08inv_3/sequential/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????	2#
!inv_3/sequential/conv2d_1/BiasAdd?
inv_3/reshape/ShapeShape*inv_3/sequential/conv2d_1/BiasAdd:output:0*
T0*
_output_shapes
:2
inv_3/reshape/Shape?
!inv_3/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2#
!inv_3/reshape/strided_slice/stack?
#inv_3/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#inv_3/reshape/strided_slice/stack_1?
#inv_3/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#inv_3/reshape/strided_slice/stack_2?
inv_3/reshape/strided_sliceStridedSliceinv_3/reshape/Shape:output:0*inv_3/reshape/strided_slice/stack:output:0,inv_3/reshape/strided_slice/stack_1:output:0,inv_3/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
inv_3/reshape/strided_slice?
inv_3/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
inv_3/reshape/Reshape/shape/1?
inv_3/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
inv_3/reshape/Reshape/shape/2?
inv_3/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :	2
inv_3/reshape/Reshape/shape/3?
inv_3/reshape/Reshape/shape/4Const*
_output_shapes
: *
dtype0*
value	B :2
inv_3/reshape/Reshape/shape/4?
inv_3/reshape/Reshape/shape/5Const*
_output_shapes
: *
dtype0*
value	B :2
inv_3/reshape/Reshape/shape/5?
inv_3/reshape/Reshape/shapePack$inv_3/reshape/strided_slice:output:0&inv_3/reshape/Reshape/shape/1:output:0&inv_3/reshape/Reshape/shape/2:output:0&inv_3/reshape/Reshape/shape/3:output:0&inv_3/reshape/Reshape/shape/4:output:0&inv_3/reshape/Reshape/shape/5:output:0*
N*
T0*
_output_shapes
:2
inv_3/reshape/Reshape/shape?
inv_3/reshape/ReshapeReshape*inv_3/sequential/conv2d_1/BiasAdd:output:0$inv_3/reshape/Reshape/shape:output:0*
T0*7
_output_shapes%
#:!?????????	2
inv_3/reshape/Reshape?
inv_3/ExtractImagePatchesExtractImagePatches max_pooling2d_1/MaxPool:output:0*
T0*/
_output_shapes
:?????????6*
ksizes
*
paddingSAME*
rates
*
strides
2
inv_3/ExtractImagePatches?
inv_3/reshape_1/ShapeShape#inv_3/ExtractImagePatches:patches:0*
T0*
_output_shapes
:2
inv_3/reshape_1/Shape?
#inv_3/reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#inv_3/reshape_1/strided_slice/stack?
%inv_3/reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%inv_3/reshape_1/strided_slice/stack_1?
%inv_3/reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%inv_3/reshape_1/strided_slice/stack_2?
inv_3/reshape_1/strided_sliceStridedSliceinv_3/reshape_1/Shape:output:0,inv_3/reshape_1/strided_slice/stack:output:0.inv_3/reshape_1/strided_slice/stack_1:output:0.inv_3/reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
inv_3/reshape_1/strided_slice?
inv_3/reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2!
inv_3/reshape_1/Reshape/shape/1?
inv_3/reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2!
inv_3/reshape_1/Reshape/shape/2?
inv_3/reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :	2!
inv_3/reshape_1/Reshape/shape/3?
inv_3/reshape_1/Reshape/shape/4Const*
_output_shapes
: *
dtype0*
value	B :2!
inv_3/reshape_1/Reshape/shape/4?
inv_3/reshape_1/Reshape/shape/5Const*
_output_shapes
: *
dtype0*
value	B :2!
inv_3/reshape_1/Reshape/shape/5?
inv_3/reshape_1/Reshape/shapePack&inv_3/reshape_1/strided_slice:output:0(inv_3/reshape_1/Reshape/shape/1:output:0(inv_3/reshape_1/Reshape/shape/2:output:0(inv_3/reshape_1/Reshape/shape/3:output:0(inv_3/reshape_1/Reshape/shape/4:output:0(inv_3/reshape_1/Reshape/shape/5:output:0*
N*
T0*
_output_shapes
:2
inv_3/reshape_1/Reshape/shape?
inv_3/reshape_1/ReshapeReshape#inv_3/ExtractImagePatches:patches:0&inv_3/reshape_1/Reshape/shape:output:0*
T0*7
_output_shapes%
#:!?????????	2
inv_3/reshape_1/Reshape?
	inv_3/MulMulinv_3/reshape/Reshape:output:0 inv_3/reshape_1/Reshape:output:0*
T0*7
_output_shapes%
#:!?????????	2
	inv_3/Mul|
inv_3/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
inv_3/Sum/reduction_indices?
	inv_3/SumSuminv_3/Mul:z:0$inv_3/Sum/reduction_indices:output:0*
T0*3
_output_shapes!
:?????????2
	inv_3/Sump
inv_3/reshape_2/ShapeShapeinv_3/Sum:output:0*
T0*
_output_shapes
:2
inv_3/reshape_2/Shape?
#inv_3/reshape_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#inv_3/reshape_2/strided_slice/stack?
%inv_3/reshape_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%inv_3/reshape_2/strided_slice/stack_1?
%inv_3/reshape_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%inv_3/reshape_2/strided_slice/stack_2?
inv_3/reshape_2/strided_sliceStridedSliceinv_3/reshape_2/Shape:output:0,inv_3/reshape_2/strided_slice/stack:output:0.inv_3/reshape_2/strided_slice/stack_1:output:0.inv_3/reshape_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
inv_3/reshape_2/strided_slice?
inv_3/reshape_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2!
inv_3/reshape_2/Reshape/shape/1?
inv_3/reshape_2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2!
inv_3/reshape_2/Reshape/shape/2?
inv_3/reshape_2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2!
inv_3/reshape_2/Reshape/shape/3?
inv_3/reshape_2/Reshape/shapePack&inv_3/reshape_2/strided_slice:output:0(inv_3/reshape_2/Reshape/shape/1:output:0(inv_3/reshape_2/Reshape/shape/2:output:0(inv_3/reshape_2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
inv_3/reshape_2/Reshape/shape?
inv_3/reshape_2/ReshapeReshapeinv_3/Sum:output:0&inv_3/reshape_2/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????2
inv_3/reshape_2/Reshape?
re_lu_3/ReluRelu inv_3/reshape_2/Reshape:output:0*
T0*/
_output_shapes
:?????????2
re_lu_3/Reluo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????`  2
flatten/Const?
flatten/ReshapeReshapere_lu_3/Relu:activations:0flatten/Const:output:0*
T0*(
_output_shapes
:??????????$2
flatten/Reshape?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	?$@*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense/BiasAddj

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2

dense/Relu?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/BiasAddy
dense_1/SoftmaxSoftmaxdense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_1/Softmaxt
IdentityIdentitydense_1/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOpE^inv_1/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpG^inv_1/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_14^inv_1/sequential/batch_normalization/ReadVariableOp6^inv_1/sequential/batch_normalization/ReadVariableOp_1/^inv_1/sequential/conv2d/BiasAdd/ReadVariableOp.^inv_1/sequential/conv2d/Conv2D/ReadVariableOp1^inv_1/sequential/conv2d_1/BiasAdd/ReadVariableOp0^inv_1/sequential/conv2d_1/Conv2D/ReadVariableOpF^inv_21/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpH^inv_21/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_15^inv_21/sequential/batch_normalization/ReadVariableOp7^inv_21/sequential/batch_normalization/ReadVariableOp_10^inv_21/sequential/conv2d/BiasAdd/ReadVariableOp/^inv_21/sequential/conv2d/Conv2D/ReadVariableOp2^inv_21/sequential/conv2d_1/BiasAdd/ReadVariableOp1^inv_21/sequential/conv2d_1/Conv2D/ReadVariableOpF^inv_22/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpH^inv_22/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_15^inv_22/sequential/batch_normalization/ReadVariableOp7^inv_22/sequential/batch_normalization/ReadVariableOp_10^inv_22/sequential/conv2d/BiasAdd/ReadVariableOp/^inv_22/sequential/conv2d/Conv2D/ReadVariableOp2^inv_22/sequential/conv2d_1/BiasAdd/ReadVariableOp1^inv_22/sequential/conv2d_1/Conv2D/ReadVariableOpE^inv_3/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpG^inv_3/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_14^inv_3/sequential/batch_normalization/ReadVariableOp6^inv_3/sequential/batch_normalization/ReadVariableOp_1/^inv_3/sequential/conv2d/BiasAdd/ReadVariableOp.^inv_3/sequential/conv2d/Conv2D/ReadVariableOp1^inv_3/sequential/conv2d_1/BiasAdd/ReadVariableOp0^inv_3/sequential/conv2d_1/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*x
_input_shapesg
e:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2?
Dinv_1/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpDinv_1/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp2?
Finv_1/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1Finv_1/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_12j
3inv_1/sequential/batch_normalization/ReadVariableOp3inv_1/sequential/batch_normalization/ReadVariableOp2n
5inv_1/sequential/batch_normalization/ReadVariableOp_15inv_1/sequential/batch_normalization/ReadVariableOp_12`
.inv_1/sequential/conv2d/BiasAdd/ReadVariableOp.inv_1/sequential/conv2d/BiasAdd/ReadVariableOp2^
-inv_1/sequential/conv2d/Conv2D/ReadVariableOp-inv_1/sequential/conv2d/Conv2D/ReadVariableOp2d
0inv_1/sequential/conv2d_1/BiasAdd/ReadVariableOp0inv_1/sequential/conv2d_1/BiasAdd/ReadVariableOp2b
/inv_1/sequential/conv2d_1/Conv2D/ReadVariableOp/inv_1/sequential/conv2d_1/Conv2D/ReadVariableOp2?
Einv_21/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpEinv_21/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp2?
Ginv_21/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1Ginv_21/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_12l
4inv_21/sequential/batch_normalization/ReadVariableOp4inv_21/sequential/batch_normalization/ReadVariableOp2p
6inv_21/sequential/batch_normalization/ReadVariableOp_16inv_21/sequential/batch_normalization/ReadVariableOp_12b
/inv_21/sequential/conv2d/BiasAdd/ReadVariableOp/inv_21/sequential/conv2d/BiasAdd/ReadVariableOp2`
.inv_21/sequential/conv2d/Conv2D/ReadVariableOp.inv_21/sequential/conv2d/Conv2D/ReadVariableOp2f
1inv_21/sequential/conv2d_1/BiasAdd/ReadVariableOp1inv_21/sequential/conv2d_1/BiasAdd/ReadVariableOp2d
0inv_21/sequential/conv2d_1/Conv2D/ReadVariableOp0inv_21/sequential/conv2d_1/Conv2D/ReadVariableOp2?
Einv_22/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpEinv_22/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp2?
Ginv_22/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1Ginv_22/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_12l
4inv_22/sequential/batch_normalization/ReadVariableOp4inv_22/sequential/batch_normalization/ReadVariableOp2p
6inv_22/sequential/batch_normalization/ReadVariableOp_16inv_22/sequential/batch_normalization/ReadVariableOp_12b
/inv_22/sequential/conv2d/BiasAdd/ReadVariableOp/inv_22/sequential/conv2d/BiasAdd/ReadVariableOp2`
.inv_22/sequential/conv2d/Conv2D/ReadVariableOp.inv_22/sequential/conv2d/Conv2D/ReadVariableOp2f
1inv_22/sequential/conv2d_1/BiasAdd/ReadVariableOp1inv_22/sequential/conv2d_1/BiasAdd/ReadVariableOp2d
0inv_22/sequential/conv2d_1/Conv2D/ReadVariableOp0inv_22/sequential/conv2d_1/Conv2D/ReadVariableOp2?
Dinv_3/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpDinv_3/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp2?
Finv_3/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1Finv_3/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_12j
3inv_3/sequential/batch_normalization/ReadVariableOp3inv_3/sequential/batch_normalization/ReadVariableOp2n
5inv_3/sequential/batch_normalization/ReadVariableOp_15inv_3/sequential/batch_normalization/ReadVariableOp_12`
.inv_3/sequential/conv2d/BiasAdd/ReadVariableOp.inv_3/sequential/conv2d/BiasAdd/ReadVariableOp2^
-inv_3/sequential/conv2d/Conv2D/ReadVariableOp-inv_3/sequential/conv2d/Conv2D/ReadVariableOp2d
0inv_3/sequential/conv2d_1/BiasAdd/ReadVariableOp0inv_3/sequential/conv2d_1/BiasAdd/ReadVariableOp2b
/inv_3/sequential/conv2d_1/Conv2D/ReadVariableOp/inv_3/sequential/conv2d_1/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?

?
&__inference_inv_1_layer_call_fn_306182
x!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5:	
	unknown_6:	
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_inv_1_layer_call_and_return_conditional_losses_3046572
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:???????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
1
_output_shapes
:???????????

_user_specified_namex
?
?
F__inference_sequential_layer_call_and_return_conditional_losses_302247
conv2d_input'
conv2d_302226:
conv2d_302228:(
batch_normalization_302231:(
batch_normalization_302233:(
batch_normalization_302235:(
batch_normalization_302237:)
conv2d_1_302241:	
conv2d_1_302243:	
identity??+batch_normalization/StatefulPartitionedCall?conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallconv2d_inputconv2d_302226conv2d_302228*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_3019902 
conv2d/StatefulPartitionedCall?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_302231batch_normalization_302233batch_normalization_302235batch_normalization_302237*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_3020132-
+batch_normalization/StatefulPartitionedCall?
re_lu/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_re_lu_layer_call_and_return_conditional_losses_3020282
re_lu/PartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallre_lu/PartitionedCall:output:0conv2d_1_302241conv2d_1_302243*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_3020402"
 conv2d_1/StatefulPartitionedCall?
IdentityIdentity)conv2d_1/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????	2

Identity?
NoOpNoOp,^batch_normalization/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:???????????: : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall:_ [
1
_output_shapes
:???????????
&
_user_specified_nameconv2d_input
?	
?
4__inference_batch_normalization_layer_call_fn_307425

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_3019132
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_layer_call_and_return_conditional_losses_302927

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????88:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????882

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????88: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????88
 
_user_specified_nameinputs
?/
?
F__inference_sequential_layer_call_and_return_conditional_losses_307266

inputs?
%conv2d_conv2d_readvariableop_resource:4
&conv2d_biasadd_readvariableop_resource:9
+batch_normalization_readvariableop_resource:;
-batch_normalization_readvariableop_1_resource:J
<batch_normalization_fusedbatchnormv3_readvariableop_resource:L
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource:A
'conv2d_1_conv2d_readvariableop_resource:	6
(conv2d_1_biasadd_readvariableop_resource:	
identity??"batch_normalization/AssignNewValue?$batch_normalization/AssignNewValue_1?3batch_normalization/FusedBatchNormV3/ReadVariableOp?5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?"batch_normalization/ReadVariableOp?$batch_normalization/ReadVariableOp_1?conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d/BiasAdd?
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype02$
"batch_normalization/ReadVariableOp?
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype02&
$batch_normalization/ReadVariableOp_1?
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOp?
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d/BiasAdd:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2&
$batch_normalization/FusedBatchNormV3?
"batch_normalization/AssignNewValueAssignVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource1batch_normalization/FusedBatchNormV3:batch_mean:04^batch_normalization/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02$
"batch_normalization/AssignNewValue?
$batch_normalization/AssignNewValue_1AssignVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource5batch_normalization/FusedBatchNormV3:batch_variance:06^batch_normalization/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02&
$batch_normalization/AssignNewValue_1?

re_lu/ReluRelu(batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????2

re_lu/Relu?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:	*
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2Dre_lu/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????	*
paddingVALID*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????	2
conv2d_1/BiasAdd|
IdentityIdentityconv2d_1/BiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????	2

Identity?
NoOpNoOp#^batch_normalization/AssignNewValue%^batch_normalization/AssignNewValue_14^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_1^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????: : : : : : : : 2H
"batch_normalization/AssignNewValue"batch_normalization/AssignNewValue2L
$batch_normalization/AssignNewValue_1$batch_normalization/AssignNewValue_12j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?h
?	
A__inference_inv_1_layer_call_and_return_conditional_losses_306140
xJ
0sequential_conv2d_conv2d_readvariableop_resource:?
1sequential_conv2d_biasadd_readvariableop_resource:D
6sequential_batch_normalization_readvariableop_resource:F
8sequential_batch_normalization_readvariableop_1_resource:U
Gsequential_batch_normalization_fusedbatchnormv3_readvariableop_resource:W
Isequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource:L
2sequential_conv2d_1_conv2d_readvariableop_resource:	A
3sequential_conv2d_1_biasadd_readvariableop_resource:	

identity_1??-sequential/batch_normalization/AssignNewValue?/sequential/batch_normalization/AssignNewValue_1?>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp?@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?-sequential/batch_normalization/ReadVariableOp?/sequential/batch_normalization/ReadVariableOp_1?(sequential/conv2d/BiasAdd/ReadVariableOp?'sequential/conv2d/Conv2D/ReadVariableOp?*sequential/conv2d_1/BiasAdd/ReadVariableOp?)sequential/conv2d_1/Conv2D/ReadVariableOp_
IdentityIdentityx*
T0*1
_output_shapes
:???????????2

Identity?
'sequential/conv2d/Conv2D/ReadVariableOpReadVariableOp0sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02)
'sequential/conv2d/Conv2D/ReadVariableOp?
sequential/conv2d/Conv2DConv2DIdentity:output:0/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
sequential/conv2d/Conv2D?
(sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOp1sequential_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(sequential/conv2d/BiasAdd/ReadVariableOp?
sequential/conv2d/BiasAddBiasAdd!sequential/conv2d/Conv2D:output:00sequential/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
sequential/conv2d/BiasAdd?
-sequential/batch_normalization/ReadVariableOpReadVariableOp6sequential_batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential/batch_normalization/ReadVariableOp?
/sequential/batch_normalization/ReadVariableOp_1ReadVariableOp8sequential_batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype021
/sequential/batch_normalization/ReadVariableOp_1?
>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpGsequential_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02@
>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp?
@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIsequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02B
@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?
/sequential/batch_normalization/FusedBatchNormV3FusedBatchNormV3"sequential/conv2d/BiasAdd:output:05sequential/batch_normalization/ReadVariableOp:value:07sequential/batch_normalization/ReadVariableOp_1:value:0Fsequential/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Hsequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<21
/sequential/batch_normalization/FusedBatchNormV3?
-sequential/batch_normalization/AssignNewValueAssignVariableOpGsequential_batch_normalization_fusedbatchnormv3_readvariableop_resource<sequential/batch_normalization/FusedBatchNormV3:batch_mean:0?^sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02/
-sequential/batch_normalization/AssignNewValue?
/sequential/batch_normalization/AssignNewValue_1AssignVariableOpIsequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource@sequential/batch_normalization/FusedBatchNormV3:batch_variance:0A^sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype021
/sequential/batch_normalization/AssignNewValue_1?
sequential/re_lu/ReluRelu3sequential/batch_normalization/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????2
sequential/re_lu/Relu?
)sequential/conv2d_1/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:	*
dtype02+
)sequential/conv2d_1/Conv2D/ReadVariableOp?
sequential/conv2d_1/Conv2DConv2D#sequential/re_lu/Relu:activations:01sequential/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????	*
paddingVALID*
strides
2
sequential/conv2d_1/Conv2D?
*sequential/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02,
*sequential/conv2d_1/BiasAdd/ReadVariableOp?
sequential/conv2d_1/BiasAddBiasAdd#sequential/conv2d_1/Conv2D:output:02sequential/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????	2
sequential/conv2d_1/BiasAddr
reshape/ShapeShape$sequential/conv2d_1/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape/Shape?
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack?
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1?
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2?
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_sliceu
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :?2
reshape/Reshape/shape/1u
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :?2
reshape/Reshape/shape/2t
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :	2
reshape/Reshape/shape/3t
reshape/Reshape/shape/4Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/4t
reshape/Reshape/shape/5Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/5?
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0 reshape/Reshape/shape/4:output:0 reshape/Reshape/shape/5:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape?
reshape/ReshapeReshape$sequential/conv2d_1/BiasAdd:output:0reshape/Reshape/shape:output:0*
T0*9
_output_shapes'
%:#???????????	2
reshape/Reshape?
ExtractImagePatchesExtractImagePatchesx*
T0*1
_output_shapes
:???????????*
ksizes
*
paddingSAME*
rates
*
strides
2
ExtractImagePatcheso
reshape_1/ShapeShapeExtractImagePatches:patches:0*
T0*
_output_shapes
:2
reshape_1/Shape?
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_1/strided_slice/stack?
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_1?
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_2?
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_1/strided_slicey
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :?2
reshape_1/Reshape/shape/1y
reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :?2
reshape_1/Reshape/shape/2x
reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :	2
reshape_1/Reshape/shape/3x
reshape_1/Reshape/shape/4Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/4x
reshape_1/Reshape/shape/5Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/5?
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0"reshape_1/Reshape/shape/2:output:0"reshape_1/Reshape/shape/3:output:0"reshape_1/Reshape/shape/4:output:0"reshape_1/Reshape/shape/5:output:0*
N*
T0*
_output_shapes
:2
reshape_1/Reshape/shape?
reshape_1/ReshapeReshapeExtractImagePatches:patches:0 reshape_1/Reshape/shape:output:0*
T0*9
_output_shapes'
%:#???????????	2
reshape_1/Reshape?
MulMulreshape/Reshape:output:0reshape_1/Reshape:output:0*
T0*9
_output_shapes'
%:#???????????	2
Mulp
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indicesz
SumSumMul:z:0Sum/reduction_indices:output:0*
T0*5
_output_shapes#
!:???????????2
Sum^
reshape_2/ShapeShapeSum:output:0*
T0*
_output_shapes
:2
reshape_2/Shape?
reshape_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_2/strided_slice/stack?
reshape_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_2/strided_slice/stack_1?
reshape_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_2/strided_slice/stack_2?
reshape_2/strided_sliceStridedSlicereshape_2/Shape:output:0&reshape_2/strided_slice/stack:output:0(reshape_2/strided_slice/stack_1:output:0(reshape_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_2/strided_slicey
reshape_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :?2
reshape_2/Reshape/shape/1y
reshape_2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :?2
reshape_2/Reshape/shape/2x
reshape_2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_2/Reshape/shape/3?
reshape_2/Reshape/shapePack reshape_2/strided_slice:output:0"reshape_2/Reshape/shape/1:output:0"reshape_2/Reshape/shape/2:output:0"reshape_2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_2/Reshape/shape?
reshape_2/ReshapeReshapeSum:output:0 reshape_2/Reshape/shape:output:0*
T0*1
_output_shapes
:???????????2
reshape_2/Reshape?

Identity_1Identityreshape_2/Reshape:output:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity_1?
NoOpNoOp.^sequential/batch_normalization/AssignNewValue0^sequential/batch_normalization/AssignNewValue_1?^sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpA^sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1.^sequential/batch_normalization/ReadVariableOp0^sequential/batch_normalization/ReadVariableOp_1)^sequential/conv2d/BiasAdd/ReadVariableOp(^sequential/conv2d/Conv2D/ReadVariableOp+^sequential/conv2d_1/BiasAdd/ReadVariableOp*^sequential/conv2d_1/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:???????????: : : : : : : : 2^
-sequential/batch_normalization/AssignNewValue-sequential/batch_normalization/AssignNewValue2b
/sequential/batch_normalization/AssignNewValue_1/sequential/batch_normalization/AssignNewValue_12?
>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp2?
@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_12^
-sequential/batch_normalization/ReadVariableOp-sequential/batch_normalization/ReadVariableOp2b
/sequential/batch_normalization/ReadVariableOp_1/sequential/batch_normalization/ReadVariableOp_12T
(sequential/conv2d/BiasAdd/ReadVariableOp(sequential/conv2d/BiasAdd/ReadVariableOp2R
'sequential/conv2d/Conv2D/ReadVariableOp'sequential/conv2d/Conv2D/ReadVariableOp2X
*sequential/conv2d_1/BiasAdd/ReadVariableOp*sequential/conv2d_1/BiasAdd/ReadVariableOp2V
)sequential/conv2d_1/Conv2D/ReadVariableOp)sequential/conv2d_1/Conv2D/ReadVariableOp:T P
1
_output_shapes
:???????????

_user_specified_namex
?
?
O__inference_batch_normalization_layer_call_and_return_conditional_losses_302481

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????88:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????882

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????88: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????88
 
_user_specified_nameinputs
?
X
,__inference_concatenate_layer_call_fn_306609
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????88* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_3039362
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????882

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:?????????88:?????????88:Y U
/
_output_shapes
:?????????88
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:?????????88
"
_user_specified_name
inputs/1
?h
?	
A__inference_inv_3_layer_call_and_return_conditional_losses_304283
xJ
0sequential_conv2d_conv2d_readvariableop_resource:?
1sequential_conv2d_biasadd_readvariableop_resource:D
6sequential_batch_normalization_readvariableop_resource:F
8sequential_batch_normalization_readvariableop_1_resource:U
Gsequential_batch_normalization_fusedbatchnormv3_readvariableop_resource:W
Isequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource:L
2sequential_conv2d_1_conv2d_readvariableop_resource:	A
3sequential_conv2d_1_biasadd_readvariableop_resource:	

identity_1??-sequential/batch_normalization/AssignNewValue?/sequential/batch_normalization/AssignNewValue_1?>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp?@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?-sequential/batch_normalization/ReadVariableOp?/sequential/batch_normalization/ReadVariableOp_1?(sequential/conv2d/BiasAdd/ReadVariableOp?'sequential/conv2d/Conv2D/ReadVariableOp?*sequential/conv2d_1/BiasAdd/ReadVariableOp?)sequential/conv2d_1/Conv2D/ReadVariableOp]
IdentityIdentityx*
T0*/
_output_shapes
:?????????2

Identity?
'sequential/conv2d/Conv2D/ReadVariableOpReadVariableOp0sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02)
'sequential/conv2d/Conv2D/ReadVariableOp?
sequential/conv2d/Conv2DConv2DIdentity:output:0/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
sequential/conv2d/Conv2D?
(sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOp1sequential_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(sequential/conv2d/BiasAdd/ReadVariableOp?
sequential/conv2d/BiasAddBiasAdd!sequential/conv2d/Conv2D:output:00sequential/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
sequential/conv2d/BiasAdd?
-sequential/batch_normalization/ReadVariableOpReadVariableOp6sequential_batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential/batch_normalization/ReadVariableOp?
/sequential/batch_normalization/ReadVariableOp_1ReadVariableOp8sequential_batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype021
/sequential/batch_normalization/ReadVariableOp_1?
>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpGsequential_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02@
>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp?
@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIsequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02B
@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?
/sequential/batch_normalization/FusedBatchNormV3FusedBatchNormV3"sequential/conv2d/BiasAdd:output:05sequential/batch_normalization/ReadVariableOp:value:07sequential/batch_normalization/ReadVariableOp_1:value:0Fsequential/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Hsequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<21
/sequential/batch_normalization/FusedBatchNormV3?
-sequential/batch_normalization/AssignNewValueAssignVariableOpGsequential_batch_normalization_fusedbatchnormv3_readvariableop_resource<sequential/batch_normalization/FusedBatchNormV3:batch_mean:0?^sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02/
-sequential/batch_normalization/AssignNewValue?
/sequential/batch_normalization/AssignNewValue_1AssignVariableOpIsequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource@sequential/batch_normalization/FusedBatchNormV3:batch_variance:0A^sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype021
/sequential/batch_normalization/AssignNewValue_1?
sequential/re_lu/ReluRelu3sequential/batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????2
sequential/re_lu/Relu?
)sequential/conv2d_1/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:	*
dtype02+
)sequential/conv2d_1/Conv2D/ReadVariableOp?
sequential/conv2d_1/Conv2DConv2D#sequential/re_lu/Relu:activations:01sequential/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????	*
paddingVALID*
strides
2
sequential/conv2d_1/Conv2D?
*sequential/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02,
*sequential/conv2d_1/BiasAdd/ReadVariableOp?
sequential/conv2d_1/BiasAddBiasAdd#sequential/conv2d_1/Conv2D:output:02sequential/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????	2
sequential/conv2d_1/BiasAddr
reshape/ShapeShape$sequential/conv2d_1/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape/Shape?
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack?
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1?
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2?
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slicet
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/2t
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :	2
reshape/Reshape/shape/3t
reshape/Reshape/shape/4Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/4t
reshape/Reshape/shape/5Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/5?
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0 reshape/Reshape/shape/4:output:0 reshape/Reshape/shape/5:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape?
reshape/ReshapeReshape$sequential/conv2d_1/BiasAdd:output:0reshape/Reshape/shape:output:0*
T0*7
_output_shapes%
#:!?????????	2
reshape/Reshape?
ExtractImagePatchesExtractImagePatchesx*
T0*/
_output_shapes
:?????????6*
ksizes
*
paddingSAME*
rates
*
strides
2
ExtractImagePatcheso
reshape_1/ShapeShapeExtractImagePatches:patches:0*
T0*
_output_shapes
:2
reshape_1/Shape?
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_1/strided_slice/stack?
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_1?
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_2?
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_1/strided_slicex
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/1x
reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/2x
reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :	2
reshape_1/Reshape/shape/3x
reshape_1/Reshape/shape/4Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/4x
reshape_1/Reshape/shape/5Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/5?
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0"reshape_1/Reshape/shape/2:output:0"reshape_1/Reshape/shape/3:output:0"reshape_1/Reshape/shape/4:output:0"reshape_1/Reshape/shape/5:output:0*
N*
T0*
_output_shapes
:2
reshape_1/Reshape/shape?
reshape_1/ReshapeReshapeExtractImagePatches:patches:0 reshape_1/Reshape/shape:output:0*
T0*7
_output_shapes%
#:!?????????	2
reshape_1/Reshape?
MulMulreshape/Reshape:output:0reshape_1/Reshape:output:0*
T0*7
_output_shapes%
#:!?????????	2
Mulp
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indicesx
SumSumMul:z:0Sum/reduction_indices:output:0*
T0*3
_output_shapes!
:?????????2
Sum^
reshape_2/ShapeShapeSum:output:0*
T0*
_output_shapes
:2
reshape_2/Shape?
reshape_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_2/strided_slice/stack?
reshape_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_2/strided_slice/stack_1?
reshape_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_2/strided_slice/stack_2?
reshape_2/strided_sliceStridedSlicereshape_2/Shape:output:0&reshape_2/strided_slice/stack:output:0(reshape_2/strided_slice/stack_1:output:0(reshape_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_2/strided_slicex
reshape_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_2/Reshape/shape/1x
reshape_2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_2/Reshape/shape/2x
reshape_2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_2/Reshape/shape/3?
reshape_2/Reshape/shapePack reshape_2/strided_slice:output:0"reshape_2/Reshape/shape/1:output:0"reshape_2/Reshape/shape/2:output:0"reshape_2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_2/Reshape/shape?
reshape_2/ReshapeReshapeSum:output:0 reshape_2/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????2
reshape_2/Reshape?

Identity_1Identityreshape_2/Reshape:output:0^NoOp*
T0*/
_output_shapes
:?????????2

Identity_1?
NoOpNoOp.^sequential/batch_normalization/AssignNewValue0^sequential/batch_normalization/AssignNewValue_1?^sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpA^sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1.^sequential/batch_normalization/ReadVariableOp0^sequential/batch_normalization/ReadVariableOp_1)^sequential/conv2d/BiasAdd/ReadVariableOp(^sequential/conv2d/Conv2D/ReadVariableOp+^sequential/conv2d_1/BiasAdd/ReadVariableOp*^sequential/conv2d_1/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????: : : : : : : : 2^
-sequential/batch_normalization/AssignNewValue-sequential/batch_normalization/AssignNewValue2b
/sequential/batch_normalization/AssignNewValue_1/sequential/batch_normalization/AssignNewValue_12?
>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp2?
@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_12^
-sequential/batch_normalization/ReadVariableOp-sequential/batch_normalization/ReadVariableOp2b
/sequential/batch_normalization/ReadVariableOp_1/sequential/batch_normalization/ReadVariableOp_12T
(sequential/conv2d/BiasAdd/ReadVariableOp(sequential/conv2d/BiasAdd/ReadVariableOp2R
'sequential/conv2d/Conv2D/ReadVariableOp'sequential/conv2d/Conv2D/ReadVariableOp2X
*sequential/conv2d_1/BiasAdd/ReadVariableOp*sequential/conv2d_1/BiasAdd/ReadVariableOp2V
)sequential/conv2d_1/Conv2D/ReadVariableOp)sequential/conv2d_1/Conv2D/ReadVariableOp:R N
/
_output_shapes
:?????????

_user_specified_namex
?
?
O__inference_batch_normalization_layer_call_and_return_conditional_losses_302583

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????88:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????882

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????88: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????88
 
_user_specified_nameinputs
?
?
A__inference_dense_layer_call_and_return_conditional_losses_304058

inputs1
matmul_readvariableop_resource:	?$@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?$@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????$: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????$
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_layer_call_and_return_conditional_losses_307535

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
]
A__inference_re_lu_layer_call_and_return_conditional_losses_307456

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:???????????2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_layer_call_and_return_conditional_losses_302115

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1y
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:???????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?(
?
F__inference_sequential_layer_call_and_return_conditional_losses_307235

inputs?
%conv2d_conv2d_readvariableop_resource:4
&conv2d_biasadd_readvariableop_resource:9
+batch_normalization_readvariableop_resource:;
-batch_normalization_readvariableop_1_resource:J
<batch_normalization_fusedbatchnormv3_readvariableop_resource:L
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource:A
'conv2d_1_conv2d_readvariableop_resource:	6
(conv2d_1_biasadd_readvariableop_resource:	
identity??3batch_normalization/FusedBatchNormV3/ReadVariableOp?5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?"batch_normalization/ReadVariableOp?$batch_normalization/ReadVariableOp_1?conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d/BiasAdd?
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype02$
"batch_normalization/ReadVariableOp?
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype02&
$batch_normalization/ReadVariableOp_1?
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOp?
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d/BiasAdd:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
is_training( 2&
$batch_normalization/FusedBatchNormV3?

re_lu/ReluRelu(batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????2

re_lu/Relu?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:	*
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2Dre_lu/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????	*
paddingVALID*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????	2
conv2d_1/BiasAdd|
IdentityIdentityconv2d_1/BiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????	2

Identity?
NoOpNoOp4^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_1^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????: : : : : : : : 2j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
4__inference_batch_normalization_layer_call_fn_307412

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_3018692
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
4__inference_batch_normalization_layer_call_fn_307623

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????88*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_3025832
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????882

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????88: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????88
 
_user_specified_nameinputs
?
?
B__inference_conv2d_layer_call_and_return_conditional_losses_303350

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2	
BiasAdds
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_layer_call_and_return_conditional_losses_302783

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
'__inference_conv2d_layer_call_fn_307671

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????88*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_3029042
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????882

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????88: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????88
 
_user_specified_nameinputs
??
?)
I__inference_InRFNet_Model_layer_call_and_return_conditional_losses_305846

inputsP
6inv_1_sequential_conv2d_conv2d_readvariableop_resource:E
7inv_1_sequential_conv2d_biasadd_readvariableop_resource:J
<inv_1_sequential_batch_normalization_readvariableop_resource:L
>inv_1_sequential_batch_normalization_readvariableop_1_resource:[
Minv_1_sequential_batch_normalization_fusedbatchnormv3_readvariableop_resource:]
Oinv_1_sequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource:R
8inv_1_sequential_conv2d_1_conv2d_readvariableop_resource:	G
9inv_1_sequential_conv2d_1_biasadd_readvariableop_resource:	Q
7inv_22_sequential_conv2d_conv2d_readvariableop_resource:F
8inv_22_sequential_conv2d_biasadd_readvariableop_resource:K
=inv_22_sequential_batch_normalization_readvariableop_resource:M
?inv_22_sequential_batch_normalization_readvariableop_1_resource:\
Ninv_22_sequential_batch_normalization_fusedbatchnormv3_readvariableop_resource:^
Pinv_22_sequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource:S
9inv_22_sequential_conv2d_1_conv2d_readvariableop_resource:H
:inv_22_sequential_conv2d_1_biasadd_readvariableop_resource:Q
7inv_21_sequential_conv2d_conv2d_readvariableop_resource:F
8inv_21_sequential_conv2d_biasadd_readvariableop_resource:K
=inv_21_sequential_batch_normalization_readvariableop_resource:M
?inv_21_sequential_batch_normalization_readvariableop_1_resource:\
Ninv_21_sequential_batch_normalization_fusedbatchnormv3_readvariableop_resource:^
Pinv_21_sequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource:S
9inv_21_sequential_conv2d_1_conv2d_readvariableop_resource:H
:inv_21_sequential_conv2d_1_biasadd_readvariableop_resource:P
6inv_3_sequential_conv2d_conv2d_readvariableop_resource:E
7inv_3_sequential_conv2d_biasadd_readvariableop_resource:J
<inv_3_sequential_batch_normalization_readvariableop_resource:L
>inv_3_sequential_batch_normalization_readvariableop_1_resource:[
Minv_3_sequential_batch_normalization_fusedbatchnormv3_readvariableop_resource:]
Oinv_3_sequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource:R
8inv_3_sequential_conv2d_1_conv2d_readvariableop_resource:	G
9inv_3_sequential_conv2d_1_biasadd_readvariableop_resource:	7
$dense_matmul_readvariableop_resource:	?$@3
%dense_biasadd_readvariableop_resource:@8
&dense_1_matmul_readvariableop_resource:@5
'dense_1_biasadd_readvariableop_resource:
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?3inv_1/sequential/batch_normalization/AssignNewValue?5inv_1/sequential/batch_normalization/AssignNewValue_1?Dinv_1/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp?Finv_1/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?3inv_1/sequential/batch_normalization/ReadVariableOp?5inv_1/sequential/batch_normalization/ReadVariableOp_1?.inv_1/sequential/conv2d/BiasAdd/ReadVariableOp?-inv_1/sequential/conv2d/Conv2D/ReadVariableOp?0inv_1/sequential/conv2d_1/BiasAdd/ReadVariableOp?/inv_1/sequential/conv2d_1/Conv2D/ReadVariableOp?4inv_21/sequential/batch_normalization/AssignNewValue?6inv_21/sequential/batch_normalization/AssignNewValue_1?Einv_21/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp?Ginv_21/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?4inv_21/sequential/batch_normalization/ReadVariableOp?6inv_21/sequential/batch_normalization/ReadVariableOp_1?/inv_21/sequential/conv2d/BiasAdd/ReadVariableOp?.inv_21/sequential/conv2d/Conv2D/ReadVariableOp?1inv_21/sequential/conv2d_1/BiasAdd/ReadVariableOp?0inv_21/sequential/conv2d_1/Conv2D/ReadVariableOp?4inv_22/sequential/batch_normalization/AssignNewValue?6inv_22/sequential/batch_normalization/AssignNewValue_1?Einv_22/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp?Ginv_22/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?4inv_22/sequential/batch_normalization/ReadVariableOp?6inv_22/sequential/batch_normalization/ReadVariableOp_1?/inv_22/sequential/conv2d/BiasAdd/ReadVariableOp?.inv_22/sequential/conv2d/Conv2D/ReadVariableOp?1inv_22/sequential/conv2d_1/BiasAdd/ReadVariableOp?0inv_22/sequential/conv2d_1/Conv2D/ReadVariableOp?3inv_3/sequential/batch_normalization/AssignNewValue?5inv_3/sequential/batch_normalization/AssignNewValue_1?Dinv_3/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp?Finv_3/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?3inv_3/sequential/batch_normalization/ReadVariableOp?5inv_3/sequential/batch_normalization/ReadVariableOp_1?.inv_3/sequential/conv2d/BiasAdd/ReadVariableOp?-inv_3/sequential/conv2d/Conv2D/ReadVariableOp?0inv_3/sequential/conv2d_1/BiasAdd/ReadVariableOp?/inv_3/sequential/conv2d_1/Conv2D/ReadVariableOpp
inv_1/IdentityIdentityinputs*
T0*1
_output_shapes
:???????????2
inv_1/Identity?
-inv_1/sequential/conv2d/Conv2D/ReadVariableOpReadVariableOp6inv_1_sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02/
-inv_1/sequential/conv2d/Conv2D/ReadVariableOp?
inv_1/sequential/conv2d/Conv2DConv2Dinv_1/Identity:output:05inv_1/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2 
inv_1/sequential/conv2d/Conv2D?
.inv_1/sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOp7inv_1_sequential_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.inv_1/sequential/conv2d/BiasAdd/ReadVariableOp?
inv_1/sequential/conv2d/BiasAddBiasAdd'inv_1/sequential/conv2d/Conv2D:output:06inv_1/sequential/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2!
inv_1/sequential/conv2d/BiasAdd?
3inv_1/sequential/batch_normalization/ReadVariableOpReadVariableOp<inv_1_sequential_batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype025
3inv_1/sequential/batch_normalization/ReadVariableOp?
5inv_1/sequential/batch_normalization/ReadVariableOp_1ReadVariableOp>inv_1_sequential_batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype027
5inv_1/sequential/batch_normalization/ReadVariableOp_1?
Dinv_1/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpMinv_1_sequential_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02F
Dinv_1/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp?
Finv_1/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOinv_1_sequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02H
Finv_1/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?
5inv_1/sequential/batch_normalization/FusedBatchNormV3FusedBatchNormV3(inv_1/sequential/conv2d/BiasAdd:output:0;inv_1/sequential/batch_normalization/ReadVariableOp:value:0=inv_1/sequential/batch_normalization/ReadVariableOp_1:value:0Linv_1/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Ninv_1/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<27
5inv_1/sequential/batch_normalization/FusedBatchNormV3?
3inv_1/sequential/batch_normalization/AssignNewValueAssignVariableOpMinv_1_sequential_batch_normalization_fusedbatchnormv3_readvariableop_resourceBinv_1/sequential/batch_normalization/FusedBatchNormV3:batch_mean:0E^inv_1/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype025
3inv_1/sequential/batch_normalization/AssignNewValue?
5inv_1/sequential/batch_normalization/AssignNewValue_1AssignVariableOpOinv_1_sequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resourceFinv_1/sequential/batch_normalization/FusedBatchNormV3:batch_variance:0G^inv_1/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype027
5inv_1/sequential/batch_normalization/AssignNewValue_1?
inv_1/sequential/re_lu/ReluRelu9inv_1/sequential/batch_normalization/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????2
inv_1/sequential/re_lu/Relu?
/inv_1/sequential/conv2d_1/Conv2D/ReadVariableOpReadVariableOp8inv_1_sequential_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:	*
dtype021
/inv_1/sequential/conv2d_1/Conv2D/ReadVariableOp?
 inv_1/sequential/conv2d_1/Conv2DConv2D)inv_1/sequential/re_lu/Relu:activations:07inv_1/sequential/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????	*
paddingVALID*
strides
2"
 inv_1/sequential/conv2d_1/Conv2D?
0inv_1/sequential/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp9inv_1_sequential_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype022
0inv_1/sequential/conv2d_1/BiasAdd/ReadVariableOp?
!inv_1/sequential/conv2d_1/BiasAddBiasAdd)inv_1/sequential/conv2d_1/Conv2D:output:08inv_1/sequential/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????	2#
!inv_1/sequential/conv2d_1/BiasAdd?
inv_1/reshape/ShapeShape*inv_1/sequential/conv2d_1/BiasAdd:output:0*
T0*
_output_shapes
:2
inv_1/reshape/Shape?
!inv_1/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2#
!inv_1/reshape/strided_slice/stack?
#inv_1/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#inv_1/reshape/strided_slice/stack_1?
#inv_1/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#inv_1/reshape/strided_slice/stack_2?
inv_1/reshape/strided_sliceStridedSliceinv_1/reshape/Shape:output:0*inv_1/reshape/strided_slice/stack:output:0,inv_1/reshape/strided_slice/stack_1:output:0,inv_1/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
inv_1/reshape/strided_slice?
inv_1/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :?2
inv_1/reshape/Reshape/shape/1?
inv_1/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :?2
inv_1/reshape/Reshape/shape/2?
inv_1/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :	2
inv_1/reshape/Reshape/shape/3?
inv_1/reshape/Reshape/shape/4Const*
_output_shapes
: *
dtype0*
value	B :2
inv_1/reshape/Reshape/shape/4?
inv_1/reshape/Reshape/shape/5Const*
_output_shapes
: *
dtype0*
value	B :2
inv_1/reshape/Reshape/shape/5?
inv_1/reshape/Reshape/shapePack$inv_1/reshape/strided_slice:output:0&inv_1/reshape/Reshape/shape/1:output:0&inv_1/reshape/Reshape/shape/2:output:0&inv_1/reshape/Reshape/shape/3:output:0&inv_1/reshape/Reshape/shape/4:output:0&inv_1/reshape/Reshape/shape/5:output:0*
N*
T0*
_output_shapes
:2
inv_1/reshape/Reshape/shape?
inv_1/reshape/ReshapeReshape*inv_1/sequential/conv2d_1/BiasAdd:output:0$inv_1/reshape/Reshape/shape:output:0*
T0*9
_output_shapes'
%:#???????????	2
inv_1/reshape/Reshape?
inv_1/ExtractImagePatchesExtractImagePatchesinputs*
T0*1
_output_shapes
:???????????*
ksizes
*
paddingSAME*
rates
*
strides
2
inv_1/ExtractImagePatches?
inv_1/reshape_1/ShapeShape#inv_1/ExtractImagePatches:patches:0*
T0*
_output_shapes
:2
inv_1/reshape_1/Shape?
#inv_1/reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#inv_1/reshape_1/strided_slice/stack?
%inv_1/reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%inv_1/reshape_1/strided_slice/stack_1?
%inv_1/reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%inv_1/reshape_1/strided_slice/stack_2?
inv_1/reshape_1/strided_sliceStridedSliceinv_1/reshape_1/Shape:output:0,inv_1/reshape_1/strided_slice/stack:output:0.inv_1/reshape_1/strided_slice/stack_1:output:0.inv_1/reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
inv_1/reshape_1/strided_slice?
inv_1/reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :?2!
inv_1/reshape_1/Reshape/shape/1?
inv_1/reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :?2!
inv_1/reshape_1/Reshape/shape/2?
inv_1/reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :	2!
inv_1/reshape_1/Reshape/shape/3?
inv_1/reshape_1/Reshape/shape/4Const*
_output_shapes
: *
dtype0*
value	B :2!
inv_1/reshape_1/Reshape/shape/4?
inv_1/reshape_1/Reshape/shape/5Const*
_output_shapes
: *
dtype0*
value	B :2!
inv_1/reshape_1/Reshape/shape/5?
inv_1/reshape_1/Reshape/shapePack&inv_1/reshape_1/strided_slice:output:0(inv_1/reshape_1/Reshape/shape/1:output:0(inv_1/reshape_1/Reshape/shape/2:output:0(inv_1/reshape_1/Reshape/shape/3:output:0(inv_1/reshape_1/Reshape/shape/4:output:0(inv_1/reshape_1/Reshape/shape/5:output:0*
N*
T0*
_output_shapes
:2
inv_1/reshape_1/Reshape/shape?
inv_1/reshape_1/ReshapeReshape#inv_1/ExtractImagePatches:patches:0&inv_1/reshape_1/Reshape/shape:output:0*
T0*9
_output_shapes'
%:#???????????	2
inv_1/reshape_1/Reshape?
	inv_1/MulMulinv_1/reshape/Reshape:output:0 inv_1/reshape_1/Reshape:output:0*
T0*9
_output_shapes'
%:#???????????	2
	inv_1/Mul|
inv_1/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
inv_1/Sum/reduction_indices?
	inv_1/SumSuminv_1/Mul:z:0$inv_1/Sum/reduction_indices:output:0*
T0*5
_output_shapes#
!:???????????2
	inv_1/Sump
inv_1/reshape_2/ShapeShapeinv_1/Sum:output:0*
T0*
_output_shapes
:2
inv_1/reshape_2/Shape?
#inv_1/reshape_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#inv_1/reshape_2/strided_slice/stack?
%inv_1/reshape_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%inv_1/reshape_2/strided_slice/stack_1?
%inv_1/reshape_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%inv_1/reshape_2/strided_slice/stack_2?
inv_1/reshape_2/strided_sliceStridedSliceinv_1/reshape_2/Shape:output:0,inv_1/reshape_2/strided_slice/stack:output:0.inv_1/reshape_2/strided_slice/stack_1:output:0.inv_1/reshape_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
inv_1/reshape_2/strided_slice?
inv_1/reshape_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :?2!
inv_1/reshape_2/Reshape/shape/1?
inv_1/reshape_2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :?2!
inv_1/reshape_2/Reshape/shape/2?
inv_1/reshape_2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2!
inv_1/reshape_2/Reshape/shape/3?
inv_1/reshape_2/Reshape/shapePack&inv_1/reshape_2/strided_slice:output:0(inv_1/reshape_2/Reshape/shape/1:output:0(inv_1/reshape_2/Reshape/shape/2:output:0(inv_1/reshape_2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
inv_1/reshape_2/Reshape/shape?
inv_1/reshape_2/ReshapeReshapeinv_1/Sum:output:0&inv_1/reshape_2/Reshape/shape:output:0*
T0*1
_output_shapes
:???????????2
inv_1/reshape_2/Reshape~

re_lu/ReluRelu inv_1/reshape_2/Reshape:output:0*
T0*1
_output_shapes
:???????????2

re_lu/Relu?
max_pooling2d/MaxPoolMaxPoolre_lu/Relu:activations:0*/
_output_shapes
:?????????pp*
ksize
*
paddingVALID*
strides
2
max_pooling2d/MaxPool?
 inv_22/average_pooling2d/AvgPoolAvgPoolmax_pooling2d/MaxPool:output:0*
T0*/
_output_shapes
:?????????88*
ksize
*
paddingSAME*
strides
2"
 inv_22/average_pooling2d/AvgPool?
.inv_22/sequential/conv2d/Conv2D/ReadVariableOpReadVariableOp7inv_22_sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype020
.inv_22/sequential/conv2d/Conv2D/ReadVariableOp?
inv_22/sequential/conv2d/Conv2DConv2D)inv_22/average_pooling2d/AvgPool:output:06inv_22/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????88*
paddingVALID*
strides
2!
inv_22/sequential/conv2d/Conv2D?
/inv_22/sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOp8inv_22_sequential_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/inv_22/sequential/conv2d/BiasAdd/ReadVariableOp?
 inv_22/sequential/conv2d/BiasAddBiasAdd(inv_22/sequential/conv2d/Conv2D:output:07inv_22/sequential/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????882"
 inv_22/sequential/conv2d/BiasAdd?
4inv_22/sequential/batch_normalization/ReadVariableOpReadVariableOp=inv_22_sequential_batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype026
4inv_22/sequential/batch_normalization/ReadVariableOp?
6inv_22/sequential/batch_normalization/ReadVariableOp_1ReadVariableOp?inv_22_sequential_batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype028
6inv_22/sequential/batch_normalization/ReadVariableOp_1?
Einv_22/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpNinv_22_sequential_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02G
Einv_22/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp?
Ginv_22/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpPinv_22_sequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02I
Ginv_22/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?
6inv_22/sequential/batch_normalization/FusedBatchNormV3FusedBatchNormV3)inv_22/sequential/conv2d/BiasAdd:output:0<inv_22/sequential/batch_normalization/ReadVariableOp:value:0>inv_22/sequential/batch_normalization/ReadVariableOp_1:value:0Minv_22/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Oinv_22/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????88:::::*
epsilon%o?:*
exponential_avg_factor%
?#<28
6inv_22/sequential/batch_normalization/FusedBatchNormV3?
4inv_22/sequential/batch_normalization/AssignNewValueAssignVariableOpNinv_22_sequential_batch_normalization_fusedbatchnormv3_readvariableop_resourceCinv_22/sequential/batch_normalization/FusedBatchNormV3:batch_mean:0F^inv_22/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype026
4inv_22/sequential/batch_normalization/AssignNewValue?
6inv_22/sequential/batch_normalization/AssignNewValue_1AssignVariableOpPinv_22_sequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resourceGinv_22/sequential/batch_normalization/FusedBatchNormV3:batch_variance:0H^inv_22/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype028
6inv_22/sequential/batch_normalization/AssignNewValue_1?
inv_22/sequential/re_lu/ReluRelu:inv_22/sequential/batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????882
inv_22/sequential/re_lu/Relu?
0inv_22/sequential/conv2d_1/Conv2D/ReadVariableOpReadVariableOp9inv_22_sequential_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype022
0inv_22/sequential/conv2d_1/Conv2D/ReadVariableOp?
!inv_22/sequential/conv2d_1/Conv2DConv2D*inv_22/sequential/re_lu/Relu:activations:08inv_22/sequential/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????88*
paddingVALID*
strides
2#
!inv_22/sequential/conv2d_1/Conv2D?
1inv_22/sequential/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp:inv_22_sequential_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype023
1inv_22/sequential/conv2d_1/BiasAdd/ReadVariableOp?
"inv_22/sequential/conv2d_1/BiasAddBiasAdd*inv_22/sequential/conv2d_1/Conv2D:output:09inv_22/sequential/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????882$
"inv_22/sequential/conv2d_1/BiasAdd?
inv_22/reshape/ShapeShape+inv_22/sequential/conv2d_1/BiasAdd:output:0*
T0*
_output_shapes
:2
inv_22/reshape/Shape?
"inv_22/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"inv_22/reshape/strided_slice/stack?
$inv_22/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$inv_22/reshape/strided_slice/stack_1?
$inv_22/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$inv_22/reshape/strided_slice/stack_2?
inv_22/reshape/strided_sliceStridedSliceinv_22/reshape/Shape:output:0+inv_22/reshape/strided_slice/stack:output:0-inv_22/reshape/strided_slice/stack_1:output:0-inv_22/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
inv_22/reshape/strided_slice?
inv_22/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :82 
inv_22/reshape/Reshape/shape/1?
inv_22/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :82 
inv_22/reshape/Reshape/shape/2?
inv_22/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2 
inv_22/reshape/Reshape/shape/3?
inv_22/reshape/Reshape/shape/4Const*
_output_shapes
: *
dtype0*
value	B :2 
inv_22/reshape/Reshape/shape/4?
inv_22/reshape/Reshape/shape/5Const*
_output_shapes
: *
dtype0*
value	B :2 
inv_22/reshape/Reshape/shape/5?
inv_22/reshape/Reshape/shapePack%inv_22/reshape/strided_slice:output:0'inv_22/reshape/Reshape/shape/1:output:0'inv_22/reshape/Reshape/shape/2:output:0'inv_22/reshape/Reshape/shape/3:output:0'inv_22/reshape/Reshape/shape/4:output:0'inv_22/reshape/Reshape/shape/5:output:0*
N*
T0*
_output_shapes
:2
inv_22/reshape/Reshape/shape?
inv_22/reshape/ReshapeReshape+inv_22/sequential/conv2d_1/BiasAdd:output:0%inv_22/reshape/Reshape/shape:output:0*
T0*7
_output_shapes%
#:!?????????882
inv_22/reshape/Reshape?
inv_22/ExtractImagePatchesExtractImagePatchesmax_pooling2d/MaxPool:output:0*
T0*/
_output_shapes
:?????????88	*
ksizes
*
paddingSAME*
rates
*
strides
2
inv_22/ExtractImagePatches?
inv_22/reshape_1/ShapeShape$inv_22/ExtractImagePatches:patches:0*
T0*
_output_shapes
:2
inv_22/reshape_1/Shape?
$inv_22/reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$inv_22/reshape_1/strided_slice/stack?
&inv_22/reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&inv_22/reshape_1/strided_slice/stack_1?
&inv_22/reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&inv_22/reshape_1/strided_slice/stack_2?
inv_22/reshape_1/strided_sliceStridedSliceinv_22/reshape_1/Shape:output:0-inv_22/reshape_1/strided_slice/stack:output:0/inv_22/reshape_1/strided_slice/stack_1:output:0/inv_22/reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
inv_22/reshape_1/strided_slice?
 inv_22/reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :82"
 inv_22/reshape_1/Reshape/shape/1?
 inv_22/reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :82"
 inv_22/reshape_1/Reshape/shape/2?
 inv_22/reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2"
 inv_22/reshape_1/Reshape/shape/3?
 inv_22/reshape_1/Reshape/shape/4Const*
_output_shapes
: *
dtype0*
value	B :2"
 inv_22/reshape_1/Reshape/shape/4?
 inv_22/reshape_1/Reshape/shape/5Const*
_output_shapes
: *
dtype0*
value	B :2"
 inv_22/reshape_1/Reshape/shape/5?
inv_22/reshape_1/Reshape/shapePack'inv_22/reshape_1/strided_slice:output:0)inv_22/reshape_1/Reshape/shape/1:output:0)inv_22/reshape_1/Reshape/shape/2:output:0)inv_22/reshape_1/Reshape/shape/3:output:0)inv_22/reshape_1/Reshape/shape/4:output:0)inv_22/reshape_1/Reshape/shape/5:output:0*
N*
T0*
_output_shapes
:2 
inv_22/reshape_1/Reshape/shape?
inv_22/reshape_1/ReshapeReshape$inv_22/ExtractImagePatches:patches:0'inv_22/reshape_1/Reshape/shape:output:0*
T0*7
_output_shapes%
#:!?????????882
inv_22/reshape_1/Reshape?

inv_22/MulMulinv_22/reshape/Reshape:output:0!inv_22/reshape_1/Reshape:output:0*
T0*7
_output_shapes%
#:!?????????882

inv_22/Mul~
inv_22/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
inv_22/Sum/reduction_indices?

inv_22/SumSuminv_22/Mul:z:0%inv_22/Sum/reduction_indices:output:0*
T0*3
_output_shapes!
:?????????882

inv_22/Sums
inv_22/reshape_2/ShapeShapeinv_22/Sum:output:0*
T0*
_output_shapes
:2
inv_22/reshape_2/Shape?
$inv_22/reshape_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$inv_22/reshape_2/strided_slice/stack?
&inv_22/reshape_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&inv_22/reshape_2/strided_slice/stack_1?
&inv_22/reshape_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&inv_22/reshape_2/strided_slice/stack_2?
inv_22/reshape_2/strided_sliceStridedSliceinv_22/reshape_2/Shape:output:0-inv_22/reshape_2/strided_slice/stack:output:0/inv_22/reshape_2/strided_slice/stack_1:output:0/inv_22/reshape_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
inv_22/reshape_2/strided_slice?
 inv_22/reshape_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :82"
 inv_22/reshape_2/Reshape/shape/1?
 inv_22/reshape_2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :82"
 inv_22/reshape_2/Reshape/shape/2?
 inv_22/reshape_2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2"
 inv_22/reshape_2/Reshape/shape/3?
inv_22/reshape_2/Reshape/shapePack'inv_22/reshape_2/strided_slice:output:0)inv_22/reshape_2/Reshape/shape/1:output:0)inv_22/reshape_2/Reshape/shape/2:output:0)inv_22/reshape_2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2 
inv_22/reshape_2/Reshape/shape?
inv_22/reshape_2/ReshapeReshapeinv_22/Sum:output:0'inv_22/reshape_2/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????882
inv_22/reshape_2/Reshape?
 inv_21/average_pooling2d/AvgPoolAvgPoolmax_pooling2d/MaxPool:output:0*
T0*/
_output_shapes
:?????????88*
ksize
*
paddingSAME*
strides
2"
 inv_21/average_pooling2d/AvgPool?
.inv_21/sequential/conv2d/Conv2D/ReadVariableOpReadVariableOp7inv_21_sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype020
.inv_21/sequential/conv2d/Conv2D/ReadVariableOp?
inv_21/sequential/conv2d/Conv2DConv2D)inv_21/average_pooling2d/AvgPool:output:06inv_21/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????88*
paddingVALID*
strides
2!
inv_21/sequential/conv2d/Conv2D?
/inv_21/sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOp8inv_21_sequential_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/inv_21/sequential/conv2d/BiasAdd/ReadVariableOp?
 inv_21/sequential/conv2d/BiasAddBiasAdd(inv_21/sequential/conv2d/Conv2D:output:07inv_21/sequential/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????882"
 inv_21/sequential/conv2d/BiasAdd?
4inv_21/sequential/batch_normalization/ReadVariableOpReadVariableOp=inv_21_sequential_batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype026
4inv_21/sequential/batch_normalization/ReadVariableOp?
6inv_21/sequential/batch_normalization/ReadVariableOp_1ReadVariableOp?inv_21_sequential_batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype028
6inv_21/sequential/batch_normalization/ReadVariableOp_1?
Einv_21/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpNinv_21_sequential_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02G
Einv_21/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp?
Ginv_21/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpPinv_21_sequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02I
Ginv_21/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?
6inv_21/sequential/batch_normalization/FusedBatchNormV3FusedBatchNormV3)inv_21/sequential/conv2d/BiasAdd:output:0<inv_21/sequential/batch_normalization/ReadVariableOp:value:0>inv_21/sequential/batch_normalization/ReadVariableOp_1:value:0Minv_21/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Oinv_21/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????88:::::*
epsilon%o?:*
exponential_avg_factor%
?#<28
6inv_21/sequential/batch_normalization/FusedBatchNormV3?
4inv_21/sequential/batch_normalization/AssignNewValueAssignVariableOpNinv_21_sequential_batch_normalization_fusedbatchnormv3_readvariableop_resourceCinv_21/sequential/batch_normalization/FusedBatchNormV3:batch_mean:0F^inv_21/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype026
4inv_21/sequential/batch_normalization/AssignNewValue?
6inv_21/sequential/batch_normalization/AssignNewValue_1AssignVariableOpPinv_21_sequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resourceGinv_21/sequential/batch_normalization/FusedBatchNormV3:batch_variance:0H^inv_21/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype028
6inv_21/sequential/batch_normalization/AssignNewValue_1?
inv_21/sequential/re_lu/ReluRelu:inv_21/sequential/batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????882
inv_21/sequential/re_lu/Relu?
0inv_21/sequential/conv2d_1/Conv2D/ReadVariableOpReadVariableOp9inv_21_sequential_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype022
0inv_21/sequential/conv2d_1/Conv2D/ReadVariableOp?
!inv_21/sequential/conv2d_1/Conv2DConv2D*inv_21/sequential/re_lu/Relu:activations:08inv_21/sequential/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????88*
paddingVALID*
strides
2#
!inv_21/sequential/conv2d_1/Conv2D?
1inv_21/sequential/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp:inv_21_sequential_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype023
1inv_21/sequential/conv2d_1/BiasAdd/ReadVariableOp?
"inv_21/sequential/conv2d_1/BiasAddBiasAdd*inv_21/sequential/conv2d_1/Conv2D:output:09inv_21/sequential/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????882$
"inv_21/sequential/conv2d_1/BiasAdd?
inv_21/reshape/ShapeShape+inv_21/sequential/conv2d_1/BiasAdd:output:0*
T0*
_output_shapes
:2
inv_21/reshape/Shape?
"inv_21/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"inv_21/reshape/strided_slice/stack?
$inv_21/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$inv_21/reshape/strided_slice/stack_1?
$inv_21/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$inv_21/reshape/strided_slice/stack_2?
inv_21/reshape/strided_sliceStridedSliceinv_21/reshape/Shape:output:0+inv_21/reshape/strided_slice/stack:output:0-inv_21/reshape/strided_slice/stack_1:output:0-inv_21/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
inv_21/reshape/strided_slice?
inv_21/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :82 
inv_21/reshape/Reshape/shape/1?
inv_21/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :82 
inv_21/reshape/Reshape/shape/2?
inv_21/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2 
inv_21/reshape/Reshape/shape/3?
inv_21/reshape/Reshape/shape/4Const*
_output_shapes
: *
dtype0*
value	B :2 
inv_21/reshape/Reshape/shape/4?
inv_21/reshape/Reshape/shape/5Const*
_output_shapes
: *
dtype0*
value	B :2 
inv_21/reshape/Reshape/shape/5?
inv_21/reshape/Reshape/shapePack%inv_21/reshape/strided_slice:output:0'inv_21/reshape/Reshape/shape/1:output:0'inv_21/reshape/Reshape/shape/2:output:0'inv_21/reshape/Reshape/shape/3:output:0'inv_21/reshape/Reshape/shape/4:output:0'inv_21/reshape/Reshape/shape/5:output:0*
N*
T0*
_output_shapes
:2
inv_21/reshape/Reshape/shape?
inv_21/reshape/ReshapeReshape+inv_21/sequential/conv2d_1/BiasAdd:output:0%inv_21/reshape/Reshape/shape:output:0*
T0*7
_output_shapes%
#:!?????????882
inv_21/reshape/Reshape?
inv_21/ExtractImagePatchesExtractImagePatchesmax_pooling2d/MaxPool:output:0*
T0*/
_output_shapes
:?????????88	*
ksizes
*
paddingSAME*
rates
*
strides
2
inv_21/ExtractImagePatches?
inv_21/reshape_1/ShapeShape$inv_21/ExtractImagePatches:patches:0*
T0*
_output_shapes
:2
inv_21/reshape_1/Shape?
$inv_21/reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$inv_21/reshape_1/strided_slice/stack?
&inv_21/reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&inv_21/reshape_1/strided_slice/stack_1?
&inv_21/reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&inv_21/reshape_1/strided_slice/stack_2?
inv_21/reshape_1/strided_sliceStridedSliceinv_21/reshape_1/Shape:output:0-inv_21/reshape_1/strided_slice/stack:output:0/inv_21/reshape_1/strided_slice/stack_1:output:0/inv_21/reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
inv_21/reshape_1/strided_slice?
 inv_21/reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :82"
 inv_21/reshape_1/Reshape/shape/1?
 inv_21/reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :82"
 inv_21/reshape_1/Reshape/shape/2?
 inv_21/reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2"
 inv_21/reshape_1/Reshape/shape/3?
 inv_21/reshape_1/Reshape/shape/4Const*
_output_shapes
: *
dtype0*
value	B :2"
 inv_21/reshape_1/Reshape/shape/4?
 inv_21/reshape_1/Reshape/shape/5Const*
_output_shapes
: *
dtype0*
value	B :2"
 inv_21/reshape_1/Reshape/shape/5?
inv_21/reshape_1/Reshape/shapePack'inv_21/reshape_1/strided_slice:output:0)inv_21/reshape_1/Reshape/shape/1:output:0)inv_21/reshape_1/Reshape/shape/2:output:0)inv_21/reshape_1/Reshape/shape/3:output:0)inv_21/reshape_1/Reshape/shape/4:output:0)inv_21/reshape_1/Reshape/shape/5:output:0*
N*
T0*
_output_shapes
:2 
inv_21/reshape_1/Reshape/shape?
inv_21/reshape_1/ReshapeReshape$inv_21/ExtractImagePatches:patches:0'inv_21/reshape_1/Reshape/shape:output:0*
T0*7
_output_shapes%
#:!?????????882
inv_21/reshape_1/Reshape?

inv_21/MulMulinv_21/reshape/Reshape:output:0!inv_21/reshape_1/Reshape:output:0*
T0*7
_output_shapes%
#:!?????????882

inv_21/Mul~
inv_21/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
inv_21/Sum/reduction_indices?

inv_21/SumSuminv_21/Mul:z:0%inv_21/Sum/reduction_indices:output:0*
T0*3
_output_shapes!
:?????????882

inv_21/Sums
inv_21/reshape_2/ShapeShapeinv_21/Sum:output:0*
T0*
_output_shapes
:2
inv_21/reshape_2/Shape?
$inv_21/reshape_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$inv_21/reshape_2/strided_slice/stack?
&inv_21/reshape_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&inv_21/reshape_2/strided_slice/stack_1?
&inv_21/reshape_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&inv_21/reshape_2/strided_slice/stack_2?
inv_21/reshape_2/strided_sliceStridedSliceinv_21/reshape_2/Shape:output:0-inv_21/reshape_2/strided_slice/stack:output:0/inv_21/reshape_2/strided_slice/stack_1:output:0/inv_21/reshape_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
inv_21/reshape_2/strided_slice?
 inv_21/reshape_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :82"
 inv_21/reshape_2/Reshape/shape/1?
 inv_21/reshape_2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :82"
 inv_21/reshape_2/Reshape/shape/2?
 inv_21/reshape_2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2"
 inv_21/reshape_2/Reshape/shape/3?
inv_21/reshape_2/Reshape/shapePack'inv_21/reshape_2/strided_slice:output:0)inv_21/reshape_2/Reshape/shape/1:output:0)inv_21/reshape_2/Reshape/shape/2:output:0)inv_21/reshape_2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2 
inv_21/reshape_2/Reshape/shape?
inv_21/reshape_2/ReshapeReshapeinv_21/Sum:output:0'inv_21/reshape_2/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????882
inv_21/reshape_2/Reshape?
re_lu_1/ReluRelu!inv_21/reshape_2/Reshape:output:0*
T0*/
_output_shapes
:?????????882
re_lu_1/Relu?
re_lu_2/ReluRelu!inv_22/reshape_2/Reshape:output:0*
T0*/
_output_shapes
:?????????882
re_lu_2/Relut
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis?
concatenate/concatConcatV2re_lu_1/Relu:activations:0re_lu_2/Relu:activations:0 concatenate/concat/axis:output:0*
N*
T0*/
_output_shapes
:?????????882
concatenate/concat?
max_pooling2d_1/MaxPoolMaxPoolconcatenate/concat:output:0*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2
max_pooling2d_1/MaxPool?
inv_3/IdentityIdentity max_pooling2d_1/MaxPool:output:0*
T0*/
_output_shapes
:?????????2
inv_3/Identity?
-inv_3/sequential/conv2d/Conv2D/ReadVariableOpReadVariableOp6inv_3_sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02/
-inv_3/sequential/conv2d/Conv2D/ReadVariableOp?
inv_3/sequential/conv2d/Conv2DConv2Dinv_3/Identity:output:05inv_3/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2 
inv_3/sequential/conv2d/Conv2D?
.inv_3/sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOp7inv_3_sequential_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.inv_3/sequential/conv2d/BiasAdd/ReadVariableOp?
inv_3/sequential/conv2d/BiasAddBiasAdd'inv_3/sequential/conv2d/Conv2D:output:06inv_3/sequential/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2!
inv_3/sequential/conv2d/BiasAdd?
3inv_3/sequential/batch_normalization/ReadVariableOpReadVariableOp<inv_3_sequential_batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype025
3inv_3/sequential/batch_normalization/ReadVariableOp?
5inv_3/sequential/batch_normalization/ReadVariableOp_1ReadVariableOp>inv_3_sequential_batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype027
5inv_3/sequential/batch_normalization/ReadVariableOp_1?
Dinv_3/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpMinv_3_sequential_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02F
Dinv_3/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp?
Finv_3/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOinv_3_sequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02H
Finv_3/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?
5inv_3/sequential/batch_normalization/FusedBatchNormV3FusedBatchNormV3(inv_3/sequential/conv2d/BiasAdd:output:0;inv_3/sequential/batch_normalization/ReadVariableOp:value:0=inv_3/sequential/batch_normalization/ReadVariableOp_1:value:0Linv_3/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Ninv_3/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<27
5inv_3/sequential/batch_normalization/FusedBatchNormV3?
3inv_3/sequential/batch_normalization/AssignNewValueAssignVariableOpMinv_3_sequential_batch_normalization_fusedbatchnormv3_readvariableop_resourceBinv_3/sequential/batch_normalization/FusedBatchNormV3:batch_mean:0E^inv_3/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype025
3inv_3/sequential/batch_normalization/AssignNewValue?
5inv_3/sequential/batch_normalization/AssignNewValue_1AssignVariableOpOinv_3_sequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resourceFinv_3/sequential/batch_normalization/FusedBatchNormV3:batch_variance:0G^inv_3/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype027
5inv_3/sequential/batch_normalization/AssignNewValue_1?
inv_3/sequential/re_lu/ReluRelu9inv_3/sequential/batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????2
inv_3/sequential/re_lu/Relu?
/inv_3/sequential/conv2d_1/Conv2D/ReadVariableOpReadVariableOp8inv_3_sequential_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:	*
dtype021
/inv_3/sequential/conv2d_1/Conv2D/ReadVariableOp?
 inv_3/sequential/conv2d_1/Conv2DConv2D)inv_3/sequential/re_lu/Relu:activations:07inv_3/sequential/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????	*
paddingVALID*
strides
2"
 inv_3/sequential/conv2d_1/Conv2D?
0inv_3/sequential/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp9inv_3_sequential_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype022
0inv_3/sequential/conv2d_1/BiasAdd/ReadVariableOp?
!inv_3/sequential/conv2d_1/BiasAddBiasAdd)inv_3/sequential/conv2d_1/Conv2D:output:08inv_3/sequential/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????	2#
!inv_3/sequential/conv2d_1/BiasAdd?
inv_3/reshape/ShapeShape*inv_3/sequential/conv2d_1/BiasAdd:output:0*
T0*
_output_shapes
:2
inv_3/reshape/Shape?
!inv_3/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2#
!inv_3/reshape/strided_slice/stack?
#inv_3/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#inv_3/reshape/strided_slice/stack_1?
#inv_3/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#inv_3/reshape/strided_slice/stack_2?
inv_3/reshape/strided_sliceStridedSliceinv_3/reshape/Shape:output:0*inv_3/reshape/strided_slice/stack:output:0,inv_3/reshape/strided_slice/stack_1:output:0,inv_3/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
inv_3/reshape/strided_slice?
inv_3/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
inv_3/reshape/Reshape/shape/1?
inv_3/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
inv_3/reshape/Reshape/shape/2?
inv_3/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :	2
inv_3/reshape/Reshape/shape/3?
inv_3/reshape/Reshape/shape/4Const*
_output_shapes
: *
dtype0*
value	B :2
inv_3/reshape/Reshape/shape/4?
inv_3/reshape/Reshape/shape/5Const*
_output_shapes
: *
dtype0*
value	B :2
inv_3/reshape/Reshape/shape/5?
inv_3/reshape/Reshape/shapePack$inv_3/reshape/strided_slice:output:0&inv_3/reshape/Reshape/shape/1:output:0&inv_3/reshape/Reshape/shape/2:output:0&inv_3/reshape/Reshape/shape/3:output:0&inv_3/reshape/Reshape/shape/4:output:0&inv_3/reshape/Reshape/shape/5:output:0*
N*
T0*
_output_shapes
:2
inv_3/reshape/Reshape/shape?
inv_3/reshape/ReshapeReshape*inv_3/sequential/conv2d_1/BiasAdd:output:0$inv_3/reshape/Reshape/shape:output:0*
T0*7
_output_shapes%
#:!?????????	2
inv_3/reshape/Reshape?
inv_3/ExtractImagePatchesExtractImagePatches max_pooling2d_1/MaxPool:output:0*
T0*/
_output_shapes
:?????????6*
ksizes
*
paddingSAME*
rates
*
strides
2
inv_3/ExtractImagePatches?
inv_3/reshape_1/ShapeShape#inv_3/ExtractImagePatches:patches:0*
T0*
_output_shapes
:2
inv_3/reshape_1/Shape?
#inv_3/reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#inv_3/reshape_1/strided_slice/stack?
%inv_3/reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%inv_3/reshape_1/strided_slice/stack_1?
%inv_3/reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%inv_3/reshape_1/strided_slice/stack_2?
inv_3/reshape_1/strided_sliceStridedSliceinv_3/reshape_1/Shape:output:0,inv_3/reshape_1/strided_slice/stack:output:0.inv_3/reshape_1/strided_slice/stack_1:output:0.inv_3/reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
inv_3/reshape_1/strided_slice?
inv_3/reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2!
inv_3/reshape_1/Reshape/shape/1?
inv_3/reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2!
inv_3/reshape_1/Reshape/shape/2?
inv_3/reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :	2!
inv_3/reshape_1/Reshape/shape/3?
inv_3/reshape_1/Reshape/shape/4Const*
_output_shapes
: *
dtype0*
value	B :2!
inv_3/reshape_1/Reshape/shape/4?
inv_3/reshape_1/Reshape/shape/5Const*
_output_shapes
: *
dtype0*
value	B :2!
inv_3/reshape_1/Reshape/shape/5?
inv_3/reshape_1/Reshape/shapePack&inv_3/reshape_1/strided_slice:output:0(inv_3/reshape_1/Reshape/shape/1:output:0(inv_3/reshape_1/Reshape/shape/2:output:0(inv_3/reshape_1/Reshape/shape/3:output:0(inv_3/reshape_1/Reshape/shape/4:output:0(inv_3/reshape_1/Reshape/shape/5:output:0*
N*
T0*
_output_shapes
:2
inv_3/reshape_1/Reshape/shape?
inv_3/reshape_1/ReshapeReshape#inv_3/ExtractImagePatches:patches:0&inv_3/reshape_1/Reshape/shape:output:0*
T0*7
_output_shapes%
#:!?????????	2
inv_3/reshape_1/Reshape?
	inv_3/MulMulinv_3/reshape/Reshape:output:0 inv_3/reshape_1/Reshape:output:0*
T0*7
_output_shapes%
#:!?????????	2
	inv_3/Mul|
inv_3/Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
inv_3/Sum/reduction_indices?
	inv_3/SumSuminv_3/Mul:z:0$inv_3/Sum/reduction_indices:output:0*
T0*3
_output_shapes!
:?????????2
	inv_3/Sump
inv_3/reshape_2/ShapeShapeinv_3/Sum:output:0*
T0*
_output_shapes
:2
inv_3/reshape_2/Shape?
#inv_3/reshape_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#inv_3/reshape_2/strided_slice/stack?
%inv_3/reshape_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%inv_3/reshape_2/strided_slice/stack_1?
%inv_3/reshape_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%inv_3/reshape_2/strided_slice/stack_2?
inv_3/reshape_2/strided_sliceStridedSliceinv_3/reshape_2/Shape:output:0,inv_3/reshape_2/strided_slice/stack:output:0.inv_3/reshape_2/strided_slice/stack_1:output:0.inv_3/reshape_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
inv_3/reshape_2/strided_slice?
inv_3/reshape_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2!
inv_3/reshape_2/Reshape/shape/1?
inv_3/reshape_2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2!
inv_3/reshape_2/Reshape/shape/2?
inv_3/reshape_2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2!
inv_3/reshape_2/Reshape/shape/3?
inv_3/reshape_2/Reshape/shapePack&inv_3/reshape_2/strided_slice:output:0(inv_3/reshape_2/Reshape/shape/1:output:0(inv_3/reshape_2/Reshape/shape/2:output:0(inv_3/reshape_2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
inv_3/reshape_2/Reshape/shape?
inv_3/reshape_2/ReshapeReshapeinv_3/Sum:output:0&inv_3/reshape_2/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????2
inv_3/reshape_2/Reshape?
re_lu_3/ReluRelu inv_3/reshape_2/Reshape:output:0*
T0*/
_output_shapes
:?????????2
re_lu_3/Reluo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????`  2
flatten/Const?
flatten/ReshapeReshapere_lu_3/Relu:activations:0flatten/Const:output:0*
T0*(
_output_shapes
:??????????$2
flatten/Reshape?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	?$@*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense/BiasAddj

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2

dense/Relu?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/BiasAddy
dense_1/SoftmaxSoftmaxdense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_1/Softmaxt
IdentityIdentitydense_1/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp4^inv_1/sequential/batch_normalization/AssignNewValue6^inv_1/sequential/batch_normalization/AssignNewValue_1E^inv_1/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpG^inv_1/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_14^inv_1/sequential/batch_normalization/ReadVariableOp6^inv_1/sequential/batch_normalization/ReadVariableOp_1/^inv_1/sequential/conv2d/BiasAdd/ReadVariableOp.^inv_1/sequential/conv2d/Conv2D/ReadVariableOp1^inv_1/sequential/conv2d_1/BiasAdd/ReadVariableOp0^inv_1/sequential/conv2d_1/Conv2D/ReadVariableOp5^inv_21/sequential/batch_normalization/AssignNewValue7^inv_21/sequential/batch_normalization/AssignNewValue_1F^inv_21/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpH^inv_21/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_15^inv_21/sequential/batch_normalization/ReadVariableOp7^inv_21/sequential/batch_normalization/ReadVariableOp_10^inv_21/sequential/conv2d/BiasAdd/ReadVariableOp/^inv_21/sequential/conv2d/Conv2D/ReadVariableOp2^inv_21/sequential/conv2d_1/BiasAdd/ReadVariableOp1^inv_21/sequential/conv2d_1/Conv2D/ReadVariableOp5^inv_22/sequential/batch_normalization/AssignNewValue7^inv_22/sequential/batch_normalization/AssignNewValue_1F^inv_22/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpH^inv_22/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_15^inv_22/sequential/batch_normalization/ReadVariableOp7^inv_22/sequential/batch_normalization/ReadVariableOp_10^inv_22/sequential/conv2d/BiasAdd/ReadVariableOp/^inv_22/sequential/conv2d/Conv2D/ReadVariableOp2^inv_22/sequential/conv2d_1/BiasAdd/ReadVariableOp1^inv_22/sequential/conv2d_1/Conv2D/ReadVariableOp4^inv_3/sequential/batch_normalization/AssignNewValue6^inv_3/sequential/batch_normalization/AssignNewValue_1E^inv_3/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpG^inv_3/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_14^inv_3/sequential/batch_normalization/ReadVariableOp6^inv_3/sequential/batch_normalization/ReadVariableOp_1/^inv_3/sequential/conv2d/BiasAdd/ReadVariableOp.^inv_3/sequential/conv2d/Conv2D/ReadVariableOp1^inv_3/sequential/conv2d_1/BiasAdd/ReadVariableOp0^inv_3/sequential/conv2d_1/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*x
_input_shapesg
e:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2j
3inv_1/sequential/batch_normalization/AssignNewValue3inv_1/sequential/batch_normalization/AssignNewValue2n
5inv_1/sequential/batch_normalization/AssignNewValue_15inv_1/sequential/batch_normalization/AssignNewValue_12?
Dinv_1/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpDinv_1/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp2?
Finv_1/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1Finv_1/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_12j
3inv_1/sequential/batch_normalization/ReadVariableOp3inv_1/sequential/batch_normalization/ReadVariableOp2n
5inv_1/sequential/batch_normalization/ReadVariableOp_15inv_1/sequential/batch_normalization/ReadVariableOp_12`
.inv_1/sequential/conv2d/BiasAdd/ReadVariableOp.inv_1/sequential/conv2d/BiasAdd/ReadVariableOp2^
-inv_1/sequential/conv2d/Conv2D/ReadVariableOp-inv_1/sequential/conv2d/Conv2D/ReadVariableOp2d
0inv_1/sequential/conv2d_1/BiasAdd/ReadVariableOp0inv_1/sequential/conv2d_1/BiasAdd/ReadVariableOp2b
/inv_1/sequential/conv2d_1/Conv2D/ReadVariableOp/inv_1/sequential/conv2d_1/Conv2D/ReadVariableOp2l
4inv_21/sequential/batch_normalization/AssignNewValue4inv_21/sequential/batch_normalization/AssignNewValue2p
6inv_21/sequential/batch_normalization/AssignNewValue_16inv_21/sequential/batch_normalization/AssignNewValue_12?
Einv_21/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpEinv_21/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp2?
Ginv_21/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1Ginv_21/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_12l
4inv_21/sequential/batch_normalization/ReadVariableOp4inv_21/sequential/batch_normalization/ReadVariableOp2p
6inv_21/sequential/batch_normalization/ReadVariableOp_16inv_21/sequential/batch_normalization/ReadVariableOp_12b
/inv_21/sequential/conv2d/BiasAdd/ReadVariableOp/inv_21/sequential/conv2d/BiasAdd/ReadVariableOp2`
.inv_21/sequential/conv2d/Conv2D/ReadVariableOp.inv_21/sequential/conv2d/Conv2D/ReadVariableOp2f
1inv_21/sequential/conv2d_1/BiasAdd/ReadVariableOp1inv_21/sequential/conv2d_1/BiasAdd/ReadVariableOp2d
0inv_21/sequential/conv2d_1/Conv2D/ReadVariableOp0inv_21/sequential/conv2d_1/Conv2D/ReadVariableOp2l
4inv_22/sequential/batch_normalization/AssignNewValue4inv_22/sequential/batch_normalization/AssignNewValue2p
6inv_22/sequential/batch_normalization/AssignNewValue_16inv_22/sequential/batch_normalization/AssignNewValue_12?
Einv_22/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpEinv_22/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp2?
Ginv_22/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1Ginv_22/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_12l
4inv_22/sequential/batch_normalization/ReadVariableOp4inv_22/sequential/batch_normalization/ReadVariableOp2p
6inv_22/sequential/batch_normalization/ReadVariableOp_16inv_22/sequential/batch_normalization/ReadVariableOp_12b
/inv_22/sequential/conv2d/BiasAdd/ReadVariableOp/inv_22/sequential/conv2d/BiasAdd/ReadVariableOp2`
.inv_22/sequential/conv2d/Conv2D/ReadVariableOp.inv_22/sequential/conv2d/Conv2D/ReadVariableOp2f
1inv_22/sequential/conv2d_1/BiasAdd/ReadVariableOp1inv_22/sequential/conv2d_1/BiasAdd/ReadVariableOp2d
0inv_22/sequential/conv2d_1/Conv2D/ReadVariableOp0inv_22/sequential/conv2d_1/Conv2D/ReadVariableOp2j
3inv_3/sequential/batch_normalization/AssignNewValue3inv_3/sequential/batch_normalization/AssignNewValue2n
5inv_3/sequential/batch_normalization/AssignNewValue_15inv_3/sequential/batch_normalization/AssignNewValue_12?
Dinv_3/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpDinv_3/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp2?
Finv_3/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1Finv_3/sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_12j
3inv_3/sequential/batch_normalization/ReadVariableOp3inv_3/sequential/batch_normalization/ReadVariableOp2n
5inv_3/sequential/batch_normalization/ReadVariableOp_15inv_3/sequential/batch_normalization/ReadVariableOp_12`
.inv_3/sequential/conv2d/BiasAdd/ReadVariableOp.inv_3/sequential/conv2d/BiasAdd/ReadVariableOp2^
-inv_3/sequential/conv2d/Conv2D/ReadVariableOp-inv_3/sequential/conv2d/Conv2D/ReadVariableOp2d
0inv_3/sequential/conv2d_1/BiasAdd/ReadVariableOp0inv_3/sequential/conv2d_1/BiasAdd/ReadVariableOp2b
/inv_3/sequential/conv2d_1/Conv2D/ReadVariableOp/inv_3/sequential/conv2d_1/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_layer_call_and_return_conditional_losses_302381

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
i
M__inference_average_pooling2d_layer_call_and_return_conditional_losses_306981

inputs
identity?
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingSAME*
strides
2	
AvgPool?
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
F__inference_sequential_layer_call_and_return_conditional_losses_302739
conv2d_input'
conv2d_302718:
conv2d_302720:(
batch_normalization_302723:(
batch_normalization_302725:(
batch_normalization_302727:(
batch_normalization_302729:)
conv2d_1_302733:
conv2d_1_302735:
identity??+batch_normalization/StatefulPartitionedCall?conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallconv2d_inputconv2d_302718conv2d_302720*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????88*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_3024582 
conv2d/StatefulPartitionedCall?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_302723batch_normalization_302725batch_normalization_302727batch_normalization_302729*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????88*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_3025832-
+batch_normalization/StatefulPartitionedCall?
re_lu/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????88* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_re_lu_layer_call_and_return_conditional_losses_3024962
re_lu/PartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallre_lu/PartitionedCall:output:0conv2d_1_302733conv2d_1_302735*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????88*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_3025082"
 conv2d_1/StatefulPartitionedCall?
IdentityIdentity)conv2d_1/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????882

Identity?
NoOpNoOp,^batch_normalization/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????88: : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall:] Y
/
_output_shapes
:?????????88
&
_user_specified_nameconv2d_input
?
g
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_306619

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????88:W S
/
_output_shapes
:?????????88
 
_user_specified_nameinputs
?	
?
4__inference_batch_normalization_layer_call_fn_307597

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_3023812
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
)__inference_conv2d_1_layer_call_fn_307824

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????88*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_3029542
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????882

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????88: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????88
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_layer_call_and_return_conditional_losses_303229

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
'__inference_conv2d_layer_call_fn_307499

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????88*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_3024582
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????882

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????88: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????88
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_layer_call_and_return_conditional_losses_307897

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
4__inference_batch_normalization_layer_call_fn_307967

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_3034752
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
B
&__inference_re_lu_layer_call_fn_306192

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_re_lu_layer_call_and_return_conditional_losses_3037312
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?	
?
4__inference_batch_normalization_layer_call_fn_307769

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_3028272
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_layer_call_and_return_conditional_losses_307345

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
F__inference_sequential_layer_call_and_return_conditional_losses_302715
conv2d_input'
conv2d_302694:
conv2d_302696:(
batch_normalization_302699:(
batch_normalization_302701:(
batch_normalization_302703:(
batch_normalization_302705:)
conv2d_1_302709:
conv2d_1_302711:
identity??+batch_normalization/StatefulPartitionedCall?conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallconv2d_inputconv2d_302694conv2d_302696*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????88*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_3024582 
conv2d/StatefulPartitionedCall?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_302699batch_normalization_302701batch_normalization_302703batch_normalization_302705*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????88*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_3024812-
+batch_normalization/StatefulPartitionedCall?
re_lu/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????88* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_re_lu_layer_call_and_return_conditional_losses_3024962
re_lu/PartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallre_lu/PartitionedCall:output:0conv2d_1_302709conv2d_1_302711*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????88*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_3025082"
 conv2d_1/StatefulPartitionedCall?
IdentityIdentity)conv2d_1/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????882

Identity?
NoOpNoOp,^batch_normalization/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????88: : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall:] Y
/
_output_shapes
:?????????88
&
_user_specified_nameconv2d_input
?`
?
B__inference_inv_22_layer_call_and_return_conditional_losses_303809
xJ
0sequential_conv2d_conv2d_readvariableop_resource:?
1sequential_conv2d_biasadd_readvariableop_resource:D
6sequential_batch_normalization_readvariableop_resource:F
8sequential_batch_normalization_readvariableop_1_resource:U
Gsequential_batch_normalization_fusedbatchnormv3_readvariableop_resource:W
Isequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource:L
2sequential_conv2d_1_conv2d_readvariableop_resource:A
3sequential_conv2d_1_biasadd_readvariableop_resource:
identity??>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp?@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?-sequential/batch_normalization/ReadVariableOp?/sequential/batch_normalization/ReadVariableOp_1?(sequential/conv2d/BiasAdd/ReadVariableOp?'sequential/conv2d/Conv2D/ReadVariableOp?*sequential/conv2d_1/BiasAdd/ReadVariableOp?)sequential/conv2d_1/Conv2D/ReadVariableOp?
average_pooling2d/AvgPoolAvgPoolx*
T0*/
_output_shapes
:?????????88*
ksize
*
paddingSAME*
strides
2
average_pooling2d/AvgPool?
'sequential/conv2d/Conv2D/ReadVariableOpReadVariableOp0sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02)
'sequential/conv2d/Conv2D/ReadVariableOp?
sequential/conv2d/Conv2DConv2D"average_pooling2d/AvgPool:output:0/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????88*
paddingVALID*
strides
2
sequential/conv2d/Conv2D?
(sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOp1sequential_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(sequential/conv2d/BiasAdd/ReadVariableOp?
sequential/conv2d/BiasAddBiasAdd!sequential/conv2d/Conv2D:output:00sequential/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????882
sequential/conv2d/BiasAdd?
-sequential/batch_normalization/ReadVariableOpReadVariableOp6sequential_batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential/batch_normalization/ReadVariableOp?
/sequential/batch_normalization/ReadVariableOp_1ReadVariableOp8sequential_batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype021
/sequential/batch_normalization/ReadVariableOp_1?
>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpGsequential_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02@
>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp?
@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIsequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02B
@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?
/sequential/batch_normalization/FusedBatchNormV3FusedBatchNormV3"sequential/conv2d/BiasAdd:output:05sequential/batch_normalization/ReadVariableOp:value:07sequential/batch_normalization/ReadVariableOp_1:value:0Fsequential/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Hsequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????88:::::*
epsilon%o?:*
is_training( 21
/sequential/batch_normalization/FusedBatchNormV3?
sequential/re_lu/ReluRelu3sequential/batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????882
sequential/re_lu/Relu?
)sequential/conv2d_1/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02+
)sequential/conv2d_1/Conv2D/ReadVariableOp?
sequential/conv2d_1/Conv2DConv2D#sequential/re_lu/Relu:activations:01sequential/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????88*
paddingVALID*
strides
2
sequential/conv2d_1/Conv2D?
*sequential/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*sequential/conv2d_1/BiasAdd/ReadVariableOp?
sequential/conv2d_1/BiasAddBiasAdd#sequential/conv2d_1/Conv2D:output:02sequential/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????882
sequential/conv2d_1/BiasAddr
reshape/ShapeShape$sequential/conv2d_1/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape/Shape?
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack?
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1?
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2?
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slicet
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :82
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :82
reshape/Reshape/shape/2t
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/3t
reshape/Reshape/shape/4Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/4t
reshape/Reshape/shape/5Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/5?
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0 reshape/Reshape/shape/4:output:0 reshape/Reshape/shape/5:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape?
reshape/ReshapeReshape$sequential/conv2d_1/BiasAdd:output:0reshape/Reshape/shape:output:0*
T0*7
_output_shapes%
#:!?????????882
reshape/Reshape?
ExtractImagePatchesExtractImagePatchesx*
T0*/
_output_shapes
:?????????88	*
ksizes
*
paddingSAME*
rates
*
strides
2
ExtractImagePatcheso
reshape_1/ShapeShapeExtractImagePatches:patches:0*
T0*
_output_shapes
:2
reshape_1/Shape?
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_1/strided_slice/stack?
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_1?
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_2?
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_1/strided_slicex
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :82
reshape_1/Reshape/shape/1x
reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :82
reshape_1/Reshape/shape/2x
reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/3x
reshape_1/Reshape/shape/4Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/4x
reshape_1/Reshape/shape/5Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/5?
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0"reshape_1/Reshape/shape/2:output:0"reshape_1/Reshape/shape/3:output:0"reshape_1/Reshape/shape/4:output:0"reshape_1/Reshape/shape/5:output:0*
N*
T0*
_output_shapes
:2
reshape_1/Reshape/shape?
reshape_1/ReshapeReshapeExtractImagePatches:patches:0 reshape_1/Reshape/shape:output:0*
T0*7
_output_shapes%
#:!?????????882
reshape_1/Reshape?
MulMulreshape/Reshape:output:0reshape_1/Reshape:output:0*
T0*7
_output_shapes%
#:!?????????882
Mulp
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indicesx
SumSumMul:z:0Sum/reduction_indices:output:0*
T0*3
_output_shapes!
:?????????882
Sum^
reshape_2/ShapeShapeSum:output:0*
T0*
_output_shapes
:2
reshape_2/Shape?
reshape_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_2/strided_slice/stack?
reshape_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_2/strided_slice/stack_1?
reshape_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_2/strided_slice/stack_2?
reshape_2/strided_sliceStridedSlicereshape_2/Shape:output:0&reshape_2/strided_slice/stack:output:0(reshape_2/strided_slice/stack_1:output:0(reshape_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_2/strided_slicex
reshape_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :82
reshape_2/Reshape/shape/1x
reshape_2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :82
reshape_2/Reshape/shape/2x
reshape_2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_2/Reshape/shape/3?
reshape_2/Reshape/shapePack reshape_2/strided_slice:output:0"reshape_2/Reshape/shape/1:output:0"reshape_2/Reshape/shape/2:output:0"reshape_2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_2/Reshape/shape?
reshape_2/ReshapeReshapeSum:output:0 reshape_2/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????882
reshape_2/Reshape}
IdentityIdentityreshape_2/Reshape:output:0^NoOp*
T0*/
_output_shapes
:?????????882

Identity?
NoOpNoOp?^sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpA^sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1.^sequential/batch_normalization/ReadVariableOp0^sequential/batch_normalization/ReadVariableOp_1)^sequential/conv2d/BiasAdd/ReadVariableOp(^sequential/conv2d/Conv2D/ReadVariableOp+^sequential/conv2d_1/BiasAdd/ReadVariableOp*^sequential/conv2d_1/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????pp: : : : : : : : 2?
>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp2?
@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_12^
-sequential/batch_normalization/ReadVariableOp-sequential/batch_normalization/ReadVariableOp2b
/sequential/batch_normalization/ReadVariableOp_1/sequential/batch_normalization/ReadVariableOp_12T
(sequential/conv2d/BiasAdd/ReadVariableOp(sequential/conv2d/BiasAdd/ReadVariableOp2R
'sequential/conv2d/Conv2D/ReadVariableOp'sequential/conv2d/Conv2D/ReadVariableOp2X
*sequential/conv2d_1/BiasAdd/ReadVariableOp*sequential/conv2d_1/BiasAdd/ReadVariableOp2V
)sequential/conv2d_1/Conv2D/ReadVariableOp)sequential/conv2d_1/Conv2D/ReadVariableOp:R N
/
_output_shapes
:?????????pp

_user_specified_namex
?
?
D__inference_conv2d_1_layer_call_and_return_conditional_losses_302040

inputs8
conv2d_readvariableop_resource:	-
biasadd_readvariableop_resource:	
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????	*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????	2	
BiasAddu
IdentityIdentityBiasAdd:output:0^NoOp*
T0*1
_output_shapes
:???????????	2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
]
A__inference_re_lu_layer_call_and_return_conditional_losses_302942

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????882
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????882

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????88:W S
/
_output_shapes
:?????????88
 
_user_specified_nameinputs
??
?,
__inference__traced_save_308322
file_prefix+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop8
4savev2_batch_normalization_gamma_read_readvariableop7
3savev2_batch_normalization_beta_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop.
*savev2_conv2d_kernel_1_read_readvariableop,
(savev2_conv2d_bias_1_read_readvariableop:
6savev2_batch_normalization_gamma_1_read_readvariableop9
5savev2_batch_normalization_beta_1_read_readvariableop0
,savev2_conv2d_1_kernel_1_read_readvariableop.
*savev2_conv2d_1_bias_1_read_readvariableop.
*savev2_conv2d_kernel_2_read_readvariableop,
(savev2_conv2d_bias_2_read_readvariableop:
6savev2_batch_normalization_gamma_2_read_readvariableop9
5savev2_batch_normalization_beta_2_read_readvariableop0
,savev2_conv2d_1_kernel_2_read_readvariableop.
*savev2_conv2d_1_bias_2_read_readvariableop.
*savev2_conv2d_kernel_3_read_readvariableop,
(savev2_conv2d_bias_3_read_readvariableop:
6savev2_batch_normalization_gamma_3_read_readvariableop9
5savev2_batch_normalization_beta_3_read_readvariableop0
,savev2_conv2d_1_kernel_3_read_readvariableop.
*savev2_conv2d_1_bias_3_read_readvariableop>
:savev2_batch_normalization_moving_mean_read_readvariableopB
>savev2_batch_normalization_moving_variance_read_readvariableop@
<savev2_batch_normalization_moving_mean_1_read_readvariableopD
@savev2_batch_normalization_moving_variance_1_read_readvariableop@
<savev2_batch_normalization_moving_mean_2_read_readvariableopD
@savev2_batch_normalization_moving_variance_2_read_readvariableop@
<savev2_batch_normalization_moving_mean_3_read_readvariableopD
@savev2_batch_normalization_moving_variance_3_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableop3
/savev2_adam_conv2d_kernel_m_read_readvariableop1
-savev2_adam_conv2d_bias_m_read_readvariableop?
;savev2_adam_batch_normalization_gamma_m_read_readvariableop>
:savev2_adam_batch_normalization_beta_m_read_readvariableop5
1savev2_adam_conv2d_1_kernel_m_read_readvariableop3
/savev2_adam_conv2d_1_bias_m_read_readvariableop5
1savev2_adam_conv2d_kernel_m_1_read_readvariableop3
/savev2_adam_conv2d_bias_m_1_read_readvariableopA
=savev2_adam_batch_normalization_gamma_m_1_read_readvariableop@
<savev2_adam_batch_normalization_beta_m_1_read_readvariableop7
3savev2_adam_conv2d_1_kernel_m_1_read_readvariableop5
1savev2_adam_conv2d_1_bias_m_1_read_readvariableop5
1savev2_adam_conv2d_kernel_m_2_read_readvariableop3
/savev2_adam_conv2d_bias_m_2_read_readvariableopA
=savev2_adam_batch_normalization_gamma_m_2_read_readvariableop@
<savev2_adam_batch_normalization_beta_m_2_read_readvariableop7
3savev2_adam_conv2d_1_kernel_m_2_read_readvariableop5
1savev2_adam_conv2d_1_bias_m_2_read_readvariableop5
1savev2_adam_conv2d_kernel_m_3_read_readvariableop3
/savev2_adam_conv2d_bias_m_3_read_readvariableopA
=savev2_adam_batch_normalization_gamma_m_3_read_readvariableop@
<savev2_adam_batch_normalization_beta_m_3_read_readvariableop7
3savev2_adam_conv2d_1_kernel_m_3_read_readvariableop5
1savev2_adam_conv2d_1_bias_m_3_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableop3
/savev2_adam_conv2d_kernel_v_read_readvariableop1
-savev2_adam_conv2d_bias_v_read_readvariableop?
;savev2_adam_batch_normalization_gamma_v_read_readvariableop>
:savev2_adam_batch_normalization_beta_v_read_readvariableop5
1savev2_adam_conv2d_1_kernel_v_read_readvariableop3
/savev2_adam_conv2d_1_bias_v_read_readvariableop5
1savev2_adam_conv2d_kernel_v_1_read_readvariableop3
/savev2_adam_conv2d_bias_v_1_read_readvariableopA
=savev2_adam_batch_normalization_gamma_v_1_read_readvariableop@
<savev2_adam_batch_normalization_beta_v_1_read_readvariableop7
3savev2_adam_conv2d_1_kernel_v_1_read_readvariableop5
1savev2_adam_conv2d_1_bias_v_1_read_readvariableop5
1savev2_adam_conv2d_kernel_v_2_read_readvariableop3
/savev2_adam_conv2d_bias_v_2_read_readvariableopA
=savev2_adam_batch_normalization_gamma_v_2_read_readvariableop@
<savev2_adam_batch_normalization_beta_v_2_read_readvariableop7
3savev2_adam_conv2d_1_kernel_v_2_read_readvariableop5
1savev2_adam_conv2d_1_bias_v_2_read_readvariableop5
1savev2_adam_conv2d_kernel_v_3_read_readvariableop3
/savev2_adam_conv2d_bias_v_3_read_readvariableopA
=savev2_adam_batch_normalization_gamma_v_3_read_readvariableop@
<savev2_adam_batch_normalization_beta_v_3_read_readvariableop7
3savev2_adam_conv2d_1_kernel_v_3_read_readvariableop5
1savev2_adam_conv2d_1_bias_v_3_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?5
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:f*
dtype0*?4
value?4B?4fB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/19/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/20/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/21/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/22/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/23/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:f*
dtype0*?
value?B?fB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?*
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop4savev2_batch_normalization_gamma_read_readvariableop3savev2_batch_normalization_beta_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop*savev2_conv2d_kernel_1_read_readvariableop(savev2_conv2d_bias_1_read_readvariableop6savev2_batch_normalization_gamma_1_read_readvariableop5savev2_batch_normalization_beta_1_read_readvariableop,savev2_conv2d_1_kernel_1_read_readvariableop*savev2_conv2d_1_bias_1_read_readvariableop*savev2_conv2d_kernel_2_read_readvariableop(savev2_conv2d_bias_2_read_readvariableop6savev2_batch_normalization_gamma_2_read_readvariableop5savev2_batch_normalization_beta_2_read_readvariableop,savev2_conv2d_1_kernel_2_read_readvariableop*savev2_conv2d_1_bias_2_read_readvariableop*savev2_conv2d_kernel_3_read_readvariableop(savev2_conv2d_bias_3_read_readvariableop6savev2_batch_normalization_gamma_3_read_readvariableop5savev2_batch_normalization_beta_3_read_readvariableop,savev2_conv2d_1_kernel_3_read_readvariableop*savev2_conv2d_1_bias_3_read_readvariableop:savev2_batch_normalization_moving_mean_read_readvariableop>savev2_batch_normalization_moving_variance_read_readvariableop<savev2_batch_normalization_moving_mean_1_read_readvariableop@savev2_batch_normalization_moving_variance_1_read_readvariableop<savev2_batch_normalization_moving_mean_2_read_readvariableop@savev2_batch_normalization_moving_variance_2_read_readvariableop<savev2_batch_normalization_moving_mean_3_read_readvariableop@savev2_batch_normalization_moving_variance_3_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableop/savev2_adam_conv2d_kernel_m_read_readvariableop-savev2_adam_conv2d_bias_m_read_readvariableop;savev2_adam_batch_normalization_gamma_m_read_readvariableop:savev2_adam_batch_normalization_beta_m_read_readvariableop1savev2_adam_conv2d_1_kernel_m_read_readvariableop/savev2_adam_conv2d_1_bias_m_read_readvariableop1savev2_adam_conv2d_kernel_m_1_read_readvariableop/savev2_adam_conv2d_bias_m_1_read_readvariableop=savev2_adam_batch_normalization_gamma_m_1_read_readvariableop<savev2_adam_batch_normalization_beta_m_1_read_readvariableop3savev2_adam_conv2d_1_kernel_m_1_read_readvariableop1savev2_adam_conv2d_1_bias_m_1_read_readvariableop1savev2_adam_conv2d_kernel_m_2_read_readvariableop/savev2_adam_conv2d_bias_m_2_read_readvariableop=savev2_adam_batch_normalization_gamma_m_2_read_readvariableop<savev2_adam_batch_normalization_beta_m_2_read_readvariableop3savev2_adam_conv2d_1_kernel_m_2_read_readvariableop1savev2_adam_conv2d_1_bias_m_2_read_readvariableop1savev2_adam_conv2d_kernel_m_3_read_readvariableop/savev2_adam_conv2d_bias_m_3_read_readvariableop=savev2_adam_batch_normalization_gamma_m_3_read_readvariableop<savev2_adam_batch_normalization_beta_m_3_read_readvariableop3savev2_adam_conv2d_1_kernel_m_3_read_readvariableop1savev2_adam_conv2d_1_bias_m_3_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableop/savev2_adam_conv2d_kernel_v_read_readvariableop-savev2_adam_conv2d_bias_v_read_readvariableop;savev2_adam_batch_normalization_gamma_v_read_readvariableop:savev2_adam_batch_normalization_beta_v_read_readvariableop1savev2_adam_conv2d_1_kernel_v_read_readvariableop/savev2_adam_conv2d_1_bias_v_read_readvariableop1savev2_adam_conv2d_kernel_v_1_read_readvariableop/savev2_adam_conv2d_bias_v_1_read_readvariableop=savev2_adam_batch_normalization_gamma_v_1_read_readvariableop<savev2_adam_batch_normalization_beta_v_1_read_readvariableop3savev2_adam_conv2d_1_kernel_v_1_read_readvariableop1savev2_adam_conv2d_1_bias_v_1_read_readvariableop1savev2_adam_conv2d_kernel_v_2_read_readvariableop/savev2_adam_conv2d_bias_v_2_read_readvariableop=savev2_adam_batch_normalization_gamma_v_2_read_readvariableop<savev2_adam_batch_normalization_beta_v_2_read_readvariableop3savev2_adam_conv2d_1_kernel_v_2_read_readvariableop1savev2_adam_conv2d_1_bias_v_2_read_readvariableop1savev2_adam_conv2d_kernel_v_3_read_readvariableop/savev2_adam_conv2d_bias_v_3_read_readvariableop=savev2_adam_batch_normalization_gamma_v_3_read_readvariableop<savev2_adam_batch_normalization_beta_v_3_read_readvariableop3savev2_adam_conv2d_1_kernel_v_3_read_readvariableop1savev2_adam_conv2d_1_bias_v_3_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *t
dtypesj
h2f	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1c
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :	?$@:@:@:: : : : : :::::	:	:::::::::::::::::	:	::::::::: : : : :	?$@:@:@::::::	:	:::::::::::::::::	:	:	?$@:@:@::::::	:	:::::::::::::::::	:	: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	?$@: 

_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :,
(
&
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:	: 

_output_shapes
:	:,(
&
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::, (
&
_output_shapes
:	: !

_output_shapes
:	: "

_output_shapes
:: #

_output_shapes
:: $

_output_shapes
:: %

_output_shapes
:: &

_output_shapes
:: '

_output_shapes
:: (

_output_shapes
:: )

_output_shapes
::*

_output_shapes
: :+

_output_shapes
: :,

_output_shapes
: :-

_output_shapes
: :%.!

_output_shapes
:	?$@: /

_output_shapes
:@:$0 

_output_shapes

:@: 1

_output_shapes
::,2(
&
_output_shapes
:: 3

_output_shapes
:: 4

_output_shapes
:: 5

_output_shapes
::,6(
&
_output_shapes
:	: 7

_output_shapes
:	:,8(
&
_output_shapes
:: 9

_output_shapes
:: :

_output_shapes
:: ;

_output_shapes
::,<(
&
_output_shapes
:: =

_output_shapes
::,>(
&
_output_shapes
:: ?

_output_shapes
:: @

_output_shapes
:: A

_output_shapes
::,B(
&
_output_shapes
:: C

_output_shapes
::,D(
&
_output_shapes
:: E

_output_shapes
:: F

_output_shapes
:: G

_output_shapes
::,H(
&
_output_shapes
:	: I

_output_shapes
:	:%J!

_output_shapes
:	?$@: K

_output_shapes
:@:$L 

_output_shapes

:@: M

_output_shapes
::,N(
&
_output_shapes
:: O

_output_shapes
:: P

_output_shapes
:: Q

_output_shapes
::,R(
&
_output_shapes
:	: S

_output_shapes
:	:,T(
&
_output_shapes
:: U

_output_shapes
:: V

_output_shapes
:: W

_output_shapes
::,X(
&
_output_shapes
:: Y

_output_shapes
::,Z(
&
_output_shapes
:: [

_output_shapes
:: \

_output_shapes
:: ]

_output_shapes
::,^(
&
_output_shapes
:: _

_output_shapes
::,`(
&
_output_shapes
:: a

_output_shapes
:: b

_output_shapes
:: c

_output_shapes
::,d(
&
_output_shapes
:	: e

_output_shapes
:	:f

_output_shapes
: 
?
?
'__inference_conv2d_layer_call_fn_307843

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_3033502
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
4__inference_batch_normalization_layer_call_fn_307928

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_3032292
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
B__inference_conv2d_layer_call_and_return_conditional_losses_307834

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2	
BiasAdds
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
_
C__inference_re_lu_2_layer_call_and_return_conditional_losses_303927

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????882
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????882

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????88:W S
/
_output_shapes
:?????????88
 
_user_specified_nameinputs
?
g
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_306614

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
4__inference_batch_normalization_layer_call_fn_307795

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????88*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_3030292
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????882

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????88: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????88
 
_user_specified_nameinputs
?
?
F__inference_sequential_layer_call_and_return_conditional_losses_303185
conv2d_input'
conv2d_303164:
conv2d_303166:(
batch_normalization_303169:(
batch_normalization_303171:(
batch_normalization_303173:(
batch_normalization_303175:)
conv2d_1_303179:
conv2d_1_303181:
identity??+batch_normalization/StatefulPartitionedCall?conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallconv2d_inputconv2d_303164conv2d_303166*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????88*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_3029042 
conv2d/StatefulPartitionedCall?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_303169batch_normalization_303171batch_normalization_303173batch_normalization_303175*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????88*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_3030292-
+batch_normalization/StatefulPartitionedCall?
re_lu/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????88* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_re_lu_layer_call_and_return_conditional_losses_3029422
re_lu/PartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallre_lu/PartitionedCall:output:0conv2d_1_303179conv2d_1_303181*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????88*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_3029542"
 conv2d_1/StatefulPartitionedCall?
IdentityIdentity)conv2d_1/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????882

Identity?
NoOpNoOp,^batch_normalization/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????88: : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall:] Y
/
_output_shapes
:?????????88
&
_user_specified_nameconv2d_input
?i
?	
B__inference_inv_21_layer_call_and_return_conditional_losses_306352
xJ
0sequential_conv2d_conv2d_readvariableop_resource:?
1sequential_conv2d_biasadd_readvariableop_resource:D
6sequential_batch_normalization_readvariableop_resource:F
8sequential_batch_normalization_readvariableop_1_resource:U
Gsequential_batch_normalization_fusedbatchnormv3_readvariableop_resource:W
Isequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource:L
2sequential_conv2d_1_conv2d_readvariableop_resource:A
3sequential_conv2d_1_biasadd_readvariableop_resource:
identity??-sequential/batch_normalization/AssignNewValue?/sequential/batch_normalization/AssignNewValue_1?>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp?@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?-sequential/batch_normalization/ReadVariableOp?/sequential/batch_normalization/ReadVariableOp_1?(sequential/conv2d/BiasAdd/ReadVariableOp?'sequential/conv2d/Conv2D/ReadVariableOp?*sequential/conv2d_1/BiasAdd/ReadVariableOp?)sequential/conv2d_1/Conv2D/ReadVariableOp?
average_pooling2d/AvgPoolAvgPoolx*
T0*/
_output_shapes
:?????????88*
ksize
*
paddingSAME*
strides
2
average_pooling2d/AvgPool?
'sequential/conv2d/Conv2D/ReadVariableOpReadVariableOp0sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02)
'sequential/conv2d/Conv2D/ReadVariableOp?
sequential/conv2d/Conv2DConv2D"average_pooling2d/AvgPool:output:0/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????88*
paddingVALID*
strides
2
sequential/conv2d/Conv2D?
(sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOp1sequential_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(sequential/conv2d/BiasAdd/ReadVariableOp?
sequential/conv2d/BiasAddBiasAdd!sequential/conv2d/Conv2D:output:00sequential/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????882
sequential/conv2d/BiasAdd?
-sequential/batch_normalization/ReadVariableOpReadVariableOp6sequential_batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential/batch_normalization/ReadVariableOp?
/sequential/batch_normalization/ReadVariableOp_1ReadVariableOp8sequential_batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype021
/sequential/batch_normalization/ReadVariableOp_1?
>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpGsequential_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02@
>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp?
@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIsequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02B
@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?
/sequential/batch_normalization/FusedBatchNormV3FusedBatchNormV3"sequential/conv2d/BiasAdd:output:05sequential/batch_normalization/ReadVariableOp:value:07sequential/batch_normalization/ReadVariableOp_1:value:0Fsequential/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Hsequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????88:::::*
epsilon%o?:*
exponential_avg_factor%
?#<21
/sequential/batch_normalization/FusedBatchNormV3?
-sequential/batch_normalization/AssignNewValueAssignVariableOpGsequential_batch_normalization_fusedbatchnormv3_readvariableop_resource<sequential/batch_normalization/FusedBatchNormV3:batch_mean:0?^sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02/
-sequential/batch_normalization/AssignNewValue?
/sequential/batch_normalization/AssignNewValue_1AssignVariableOpIsequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource@sequential/batch_normalization/FusedBatchNormV3:batch_variance:0A^sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype021
/sequential/batch_normalization/AssignNewValue_1?
sequential/re_lu/ReluRelu3sequential/batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????882
sequential/re_lu/Relu?
)sequential/conv2d_1/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02+
)sequential/conv2d_1/Conv2D/ReadVariableOp?
sequential/conv2d_1/Conv2DConv2D#sequential/re_lu/Relu:activations:01sequential/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????88*
paddingVALID*
strides
2
sequential/conv2d_1/Conv2D?
*sequential/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*sequential/conv2d_1/BiasAdd/ReadVariableOp?
sequential/conv2d_1/BiasAddBiasAdd#sequential/conv2d_1/Conv2D:output:02sequential/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????882
sequential/conv2d_1/BiasAddr
reshape/ShapeShape$sequential/conv2d_1/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape/Shape?
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack?
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1?
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2?
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slicet
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :82
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :82
reshape/Reshape/shape/2t
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/3t
reshape/Reshape/shape/4Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/4t
reshape/Reshape/shape/5Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/5?
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0 reshape/Reshape/shape/4:output:0 reshape/Reshape/shape/5:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape?
reshape/ReshapeReshape$sequential/conv2d_1/BiasAdd:output:0reshape/Reshape/shape:output:0*
T0*7
_output_shapes%
#:!?????????882
reshape/Reshape?
ExtractImagePatchesExtractImagePatchesx*
T0*/
_output_shapes
:?????????88	*
ksizes
*
paddingSAME*
rates
*
strides
2
ExtractImagePatcheso
reshape_1/ShapeShapeExtractImagePatches:patches:0*
T0*
_output_shapes
:2
reshape_1/Shape?
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_1/strided_slice/stack?
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_1?
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_2?
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_1/strided_slicex
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :82
reshape_1/Reshape/shape/1x
reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :82
reshape_1/Reshape/shape/2x
reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/3x
reshape_1/Reshape/shape/4Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/4x
reshape_1/Reshape/shape/5Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/5?
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0"reshape_1/Reshape/shape/2:output:0"reshape_1/Reshape/shape/3:output:0"reshape_1/Reshape/shape/4:output:0"reshape_1/Reshape/shape/5:output:0*
N*
T0*
_output_shapes
:2
reshape_1/Reshape/shape?
reshape_1/ReshapeReshapeExtractImagePatches:patches:0 reshape_1/Reshape/shape:output:0*
T0*7
_output_shapes%
#:!?????????882
reshape_1/Reshape?
MulMulreshape/Reshape:output:0reshape_1/Reshape:output:0*
T0*7
_output_shapes%
#:!?????????882
Mulp
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indicesx
SumSumMul:z:0Sum/reduction_indices:output:0*
T0*3
_output_shapes!
:?????????882
Sum^
reshape_2/ShapeShapeSum:output:0*
T0*
_output_shapes
:2
reshape_2/Shape?
reshape_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_2/strided_slice/stack?
reshape_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_2/strided_slice/stack_1?
reshape_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_2/strided_slice/stack_2?
reshape_2/strided_sliceStridedSlicereshape_2/Shape:output:0&reshape_2/strided_slice/stack:output:0(reshape_2/strided_slice/stack_1:output:0(reshape_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_2/strided_slicex
reshape_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :82
reshape_2/Reshape/shape/1x
reshape_2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :82
reshape_2/Reshape/shape/2x
reshape_2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_2/Reshape/shape/3?
reshape_2/Reshape/shapePack reshape_2/strided_slice:output:0"reshape_2/Reshape/shape/1:output:0"reshape_2/Reshape/shape/2:output:0"reshape_2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_2/Reshape/shape?
reshape_2/ReshapeReshapeSum:output:0 reshape_2/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????882
reshape_2/Reshape}
IdentityIdentityreshape_2/Reshape:output:0^NoOp*
T0*/
_output_shapes
:?????????882

Identity?
NoOpNoOp.^sequential/batch_normalization/AssignNewValue0^sequential/batch_normalization/AssignNewValue_1?^sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpA^sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1.^sequential/batch_normalization/ReadVariableOp0^sequential/batch_normalization/ReadVariableOp_1)^sequential/conv2d/BiasAdd/ReadVariableOp(^sequential/conv2d/Conv2D/ReadVariableOp+^sequential/conv2d_1/BiasAdd/ReadVariableOp*^sequential/conv2d_1/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????pp: : : : : : : : 2^
-sequential/batch_normalization/AssignNewValue-sequential/batch_normalization/AssignNewValue2b
/sequential/batch_normalization/AssignNewValue_1/sequential/batch_normalization/AssignNewValue_12?
>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp2?
@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_12^
-sequential/batch_normalization/ReadVariableOp-sequential/batch_normalization/ReadVariableOp2b
/sequential/batch_normalization/ReadVariableOp_1/sequential/batch_normalization/ReadVariableOp_12T
(sequential/conv2d/BiasAdd/ReadVariableOp(sequential/conv2d/BiasAdd/ReadVariableOp2R
'sequential/conv2d/Conv2D/ReadVariableOp'sequential/conv2d/Conv2D/ReadVariableOp2X
*sequential/conv2d_1/BiasAdd/ReadVariableOp*sequential/conv2d_1/BiasAdd/ReadVariableOp2V
)sequential/conv2d_1/Conv2D/ReadVariableOp)sequential/conv2d_1/Conv2D/ReadVariableOp:R N
/
_output_shapes
:?????????pp

_user_specified_namex
?

?
+__inference_sequential_layer_call_fn_302534
conv2d_input!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????88**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_3025152
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????882

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????88: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:?????????88
&
_user_specified_nameconv2d_input
?
?
4__inference_batch_normalization_layer_call_fn_307610

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????88*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_3024812
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????882

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????88: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????88
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_layer_call_and_return_conditional_losses_307861

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_layer_call_and_return_conditional_losses_307689

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?i
?	
B__inference_inv_21_layer_call_and_return_conditional_losses_304420
xJ
0sequential_conv2d_conv2d_readvariableop_resource:?
1sequential_conv2d_biasadd_readvariableop_resource:D
6sequential_batch_normalization_readvariableop_resource:F
8sequential_batch_normalization_readvariableop_1_resource:U
Gsequential_batch_normalization_fusedbatchnormv3_readvariableop_resource:W
Isequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource:L
2sequential_conv2d_1_conv2d_readvariableop_resource:A
3sequential_conv2d_1_biasadd_readvariableop_resource:
identity??-sequential/batch_normalization/AssignNewValue?/sequential/batch_normalization/AssignNewValue_1?>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp?@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?-sequential/batch_normalization/ReadVariableOp?/sequential/batch_normalization/ReadVariableOp_1?(sequential/conv2d/BiasAdd/ReadVariableOp?'sequential/conv2d/Conv2D/ReadVariableOp?*sequential/conv2d_1/BiasAdd/ReadVariableOp?)sequential/conv2d_1/Conv2D/ReadVariableOp?
average_pooling2d/AvgPoolAvgPoolx*
T0*/
_output_shapes
:?????????88*
ksize
*
paddingSAME*
strides
2
average_pooling2d/AvgPool?
'sequential/conv2d/Conv2D/ReadVariableOpReadVariableOp0sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02)
'sequential/conv2d/Conv2D/ReadVariableOp?
sequential/conv2d/Conv2DConv2D"average_pooling2d/AvgPool:output:0/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????88*
paddingVALID*
strides
2
sequential/conv2d/Conv2D?
(sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOp1sequential_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(sequential/conv2d/BiasAdd/ReadVariableOp?
sequential/conv2d/BiasAddBiasAdd!sequential/conv2d/Conv2D:output:00sequential/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????882
sequential/conv2d/BiasAdd?
-sequential/batch_normalization/ReadVariableOpReadVariableOp6sequential_batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential/batch_normalization/ReadVariableOp?
/sequential/batch_normalization/ReadVariableOp_1ReadVariableOp8sequential_batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype021
/sequential/batch_normalization/ReadVariableOp_1?
>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpGsequential_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02@
>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp?
@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIsequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02B
@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?
/sequential/batch_normalization/FusedBatchNormV3FusedBatchNormV3"sequential/conv2d/BiasAdd:output:05sequential/batch_normalization/ReadVariableOp:value:07sequential/batch_normalization/ReadVariableOp_1:value:0Fsequential/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Hsequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????88:::::*
epsilon%o?:*
exponential_avg_factor%
?#<21
/sequential/batch_normalization/FusedBatchNormV3?
-sequential/batch_normalization/AssignNewValueAssignVariableOpGsequential_batch_normalization_fusedbatchnormv3_readvariableop_resource<sequential/batch_normalization/FusedBatchNormV3:batch_mean:0?^sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02/
-sequential/batch_normalization/AssignNewValue?
/sequential/batch_normalization/AssignNewValue_1AssignVariableOpIsequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource@sequential/batch_normalization/FusedBatchNormV3:batch_variance:0A^sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype021
/sequential/batch_normalization/AssignNewValue_1?
sequential/re_lu/ReluRelu3sequential/batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????882
sequential/re_lu/Relu?
)sequential/conv2d_1/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02+
)sequential/conv2d_1/Conv2D/ReadVariableOp?
sequential/conv2d_1/Conv2DConv2D#sequential/re_lu/Relu:activations:01sequential/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????88*
paddingVALID*
strides
2
sequential/conv2d_1/Conv2D?
*sequential/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*sequential/conv2d_1/BiasAdd/ReadVariableOp?
sequential/conv2d_1/BiasAddBiasAdd#sequential/conv2d_1/Conv2D:output:02sequential/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????882
sequential/conv2d_1/BiasAddr
reshape/ShapeShape$sequential/conv2d_1/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape/Shape?
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack?
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1?
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2?
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slicet
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :82
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :82
reshape/Reshape/shape/2t
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/3t
reshape/Reshape/shape/4Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/4t
reshape/Reshape/shape/5Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/5?
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0 reshape/Reshape/shape/4:output:0 reshape/Reshape/shape/5:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape?
reshape/ReshapeReshape$sequential/conv2d_1/BiasAdd:output:0reshape/Reshape/shape:output:0*
T0*7
_output_shapes%
#:!?????????882
reshape/Reshape?
ExtractImagePatchesExtractImagePatchesx*
T0*/
_output_shapes
:?????????88	*
ksizes
*
paddingSAME*
rates
*
strides
2
ExtractImagePatcheso
reshape_1/ShapeShapeExtractImagePatches:patches:0*
T0*
_output_shapes
:2
reshape_1/Shape?
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_1/strided_slice/stack?
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_1?
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_2?
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_1/strided_slicex
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :82
reshape_1/Reshape/shape/1x
reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :82
reshape_1/Reshape/shape/2x
reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/3x
reshape_1/Reshape/shape/4Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/4x
reshape_1/Reshape/shape/5Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/5?
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0"reshape_1/Reshape/shape/2:output:0"reshape_1/Reshape/shape/3:output:0"reshape_1/Reshape/shape/4:output:0"reshape_1/Reshape/shape/5:output:0*
N*
T0*
_output_shapes
:2
reshape_1/Reshape/shape?
reshape_1/ReshapeReshapeExtractImagePatches:patches:0 reshape_1/Reshape/shape:output:0*
T0*7
_output_shapes%
#:!?????????882
reshape_1/Reshape?
MulMulreshape/Reshape:output:0reshape_1/Reshape:output:0*
T0*7
_output_shapes%
#:!?????????882
Mulp
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indicesx
SumSumMul:z:0Sum/reduction_indices:output:0*
T0*3
_output_shapes!
:?????????882
Sum^
reshape_2/ShapeShapeSum:output:0*
T0*
_output_shapes
:2
reshape_2/Shape?
reshape_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_2/strided_slice/stack?
reshape_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_2/strided_slice/stack_1?
reshape_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_2/strided_slice/stack_2?
reshape_2/strided_sliceStridedSlicereshape_2/Shape:output:0&reshape_2/strided_slice/stack:output:0(reshape_2/strided_slice/stack_1:output:0(reshape_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_2/strided_slicex
reshape_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :82
reshape_2/Reshape/shape/1x
reshape_2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :82
reshape_2/Reshape/shape/2x
reshape_2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_2/Reshape/shape/3?
reshape_2/Reshape/shapePack reshape_2/strided_slice:output:0"reshape_2/Reshape/shape/1:output:0"reshape_2/Reshape/shape/2:output:0"reshape_2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_2/Reshape/shape?
reshape_2/ReshapeReshapeSum:output:0 reshape_2/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????882
reshape_2/Reshape}
IdentityIdentityreshape_2/Reshape:output:0^NoOp*
T0*/
_output_shapes
:?????????882

Identity?
NoOpNoOp.^sequential/batch_normalization/AssignNewValue0^sequential/batch_normalization/AssignNewValue_1?^sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpA^sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1.^sequential/batch_normalization/ReadVariableOp0^sequential/batch_normalization/ReadVariableOp_1)^sequential/conv2d/BiasAdd/ReadVariableOp(^sequential/conv2d/Conv2D/ReadVariableOp+^sequential/conv2d_1/BiasAdd/ReadVariableOp*^sequential/conv2d_1/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????pp: : : : : : : : 2^
-sequential/batch_normalization/AssignNewValue-sequential/batch_normalization/AssignNewValue2b
/sequential/batch_normalization/AssignNewValue_1/sequential/batch_normalization/AssignNewValue_12?
>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp2?
@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_12^
-sequential/batch_normalization/ReadVariableOp-sequential/batch_normalization/ReadVariableOp2b
/sequential/batch_normalization/ReadVariableOp_1/sequential/batch_normalization/ReadVariableOp_12T
(sequential/conv2d/BiasAdd/ReadVariableOp(sequential/conv2d/BiasAdd/ReadVariableOp2R
'sequential/conv2d/Conv2D/ReadVariableOp'sequential/conv2d/Conv2D/ReadVariableOp2X
*sequential/conv2d_1/BiasAdd/ReadVariableOp*sequential/conv2d_1/BiasAdd/ReadVariableOp2V
)sequential/conv2d_1/Conv2D/ReadVariableOp)sequential/conv2d_1/Conv2D/ReadVariableOp:R N
/
_output_shapes
:?????????pp

_user_specified_namex
?
?
O__inference_batch_normalization_layer_call_and_return_conditional_losses_303475

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
]
A__inference_re_lu_layer_call_and_return_conditional_losses_306187

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:???????????2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
F__inference_sequential_layer_call_and_return_conditional_losses_303407

inputs'
conv2d_303351:
conv2d_303353:(
batch_normalization_303374:(
batch_normalization_303376:(
batch_normalization_303378:(
batch_normalization_303380:)
conv2d_1_303401:	
conv2d_1_303403:	
identity??+batch_normalization/StatefulPartitionedCall?conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_303351conv2d_303353*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_3033502 
conv2d/StatefulPartitionedCall?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_303374batch_normalization_303376batch_normalization_303378batch_normalization_303380*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_3033732-
+batch_normalization/StatefulPartitionedCall?
re_lu/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_re_lu_layer_call_and_return_conditional_losses_3033882
re_lu/PartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallre_lu/PartitionedCall:output:0conv2d_1_303401conv2d_1_303403*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_3034002"
 conv2d_1/StatefulPartitionedCall?
IdentityIdentity)conv2d_1/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????	2

Identity?
NoOpNoOp,^batch_normalization/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????: : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
F__inference_sequential_layer_call_and_return_conditional_losses_302271
conv2d_input'
conv2d_302250:
conv2d_302252:(
batch_normalization_302255:(
batch_normalization_302257:(
batch_normalization_302259:(
batch_normalization_302261:)
conv2d_1_302265:	
conv2d_1_302267:	
identity??+batch_normalization/StatefulPartitionedCall?conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallconv2d_inputconv2d_302250conv2d_302252*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_3019902 
conv2d/StatefulPartitionedCall?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_302255batch_normalization_302257batch_normalization_302259batch_normalization_302261*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_3021152-
+batch_normalization/StatefulPartitionedCall?
re_lu/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_re_lu_layer_call_and_return_conditional_losses_3020282
re_lu/PartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallre_lu/PartitionedCall:output:0conv2d_1_302265conv2d_1_302267*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_3020402"
 conv2d_1/StatefulPartitionedCall?
IdentityIdentity)conv2d_1/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????	2

Identity?
NoOpNoOp,^batch_normalization/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:???????????: : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall:_ [
1
_output_shapes
:???????????
&
_user_specified_nameconv2d_input
?

?
+__inference_sequential_layer_call_fn_306955

inputs!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5:	
	unknown_6:	
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????	**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_3020472
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????	2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:???????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_layer_call_and_return_conditional_losses_307363

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
F__inference_sequential_layer_call_and_return_conditional_losses_302651

inputs'
conv2d_302630:
conv2d_302632:(
batch_normalization_302635:(
batch_normalization_302637:(
batch_normalization_302639:(
batch_normalization_302641:)
conv2d_1_302645:
conv2d_1_302647:
identity??+batch_normalization/StatefulPartitionedCall?conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_302630conv2d_302632*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????88*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_3024582 
conv2d/StatefulPartitionedCall?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_302635batch_normalization_302637batch_normalization_302639batch_normalization_302641*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????88*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_3025832-
+batch_normalization/StatefulPartitionedCall?
re_lu/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????88* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_re_lu_layer_call_and_return_conditional_losses_3024962
re_lu/PartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallre_lu/PartitionedCall:output:0conv2d_1_302645conv2d_1_302647*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????88*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_3025082"
 conv2d_1/StatefulPartitionedCall?
IdentityIdentity)conv2d_1/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????882

Identity?
NoOpNoOp,^batch_normalization/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????88: : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall:W S
/
_output_shapes
:?????????88
 
_user_specified_nameinputs
?

?
'__inference_inv_21_layer_call_fn_306394
x!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????88*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_inv_21_layer_call_and_return_conditional_losses_3044202
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????882

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????pp: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
/
_output_shapes
:?????????pp

_user_specified_namex
?
s
G__inference_concatenate_layer_call_and_return_conditional_losses_306603
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*/
_output_shapes
:?????????882
concatk
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:?????????882

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:?????????88:?????????88:Y U
/
_output_shapes
:?????????88
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:?????????88
"
_user_specified_name
inputs/1
?

?
'__inference_inv_22_layer_call_fn_306555
x!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????88**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_inv_22_layer_call_and_return_conditional_losses_3038092
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????882

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????pp: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
/
_output_shapes
:?????????pp

_user_specified_namex
?

?
&__inference_inv_3_layer_call_fn_306790
x!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5:	
	unknown_6:	
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_inv_3_layer_call_and_return_conditional_losses_3040142
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
/
_output_shapes
:?????????

_user_specified_namex
?
?
)__inference_conv2d_1_layer_call_fn_307480

inputs!
unknown:	
	unknown_0:	
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_3020402
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????	2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
4__inference_batch_normalization_layer_call_fn_307954

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_3033732
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
N
2__inference_average_pooling2d_layer_call_fn_307100

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_average_pooling2d_layer_call_and_return_conditional_losses_3027482
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
D__inference_conv2d_1_layer_call_and_return_conditional_losses_302508

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????88*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????882	
BiasAdds
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????882

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????88: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????88
 
_user_specified_nameinputs
?
?
B__inference_conv2d_layer_call_and_return_conditional_losses_301990

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2	
BiasAddu
IdentityIdentityBiasAdd:output:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
B__inference_conv2d_layer_call_and_return_conditional_losses_307490

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????88*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????882	
BiasAdds
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????882

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????88: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????88
 
_user_specified_nameinputs
?

?
+__inference_sequential_layer_call_fn_302691
conv2d_input!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????88*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_3026512
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????882

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????88: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:?????????88
&
_user_specified_nameconv2d_input
?
]
A__inference_re_lu_layer_call_and_return_conditional_losses_307800

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????882
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????882

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????88:W S
/
_output_shapes
:?????????88
 
_user_specified_nameinputs
?
J
.__inference_max_pooling2d_layer_call_fn_306207

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_3022802
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_layer_call_and_return_conditional_losses_302337

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
]
A__inference_re_lu_layer_call_and_return_conditional_losses_302028

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:???????????2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
)__inference_conv2d_1_layer_call_fn_307652

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????88*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_3025082
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????882

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????88: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????88
 
_user_specified_nameinputs
?_
?
A__inference_inv_3_layer_call_and_return_conditional_losses_304014
xJ
0sequential_conv2d_conv2d_readvariableop_resource:?
1sequential_conv2d_biasadd_readvariableop_resource:D
6sequential_batch_normalization_readvariableop_resource:F
8sequential_batch_normalization_readvariableop_1_resource:U
Gsequential_batch_normalization_fusedbatchnormv3_readvariableop_resource:W
Isequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource:L
2sequential_conv2d_1_conv2d_readvariableop_resource:	A
3sequential_conv2d_1_biasadd_readvariableop_resource:	

identity_1??>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp?@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?-sequential/batch_normalization/ReadVariableOp?/sequential/batch_normalization/ReadVariableOp_1?(sequential/conv2d/BiasAdd/ReadVariableOp?'sequential/conv2d/Conv2D/ReadVariableOp?*sequential/conv2d_1/BiasAdd/ReadVariableOp?)sequential/conv2d_1/Conv2D/ReadVariableOp]
IdentityIdentityx*
T0*/
_output_shapes
:?????????2

Identity?
'sequential/conv2d/Conv2D/ReadVariableOpReadVariableOp0sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02)
'sequential/conv2d/Conv2D/ReadVariableOp?
sequential/conv2d/Conv2DConv2DIdentity:output:0/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
sequential/conv2d/Conv2D?
(sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOp1sequential_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(sequential/conv2d/BiasAdd/ReadVariableOp?
sequential/conv2d/BiasAddBiasAdd!sequential/conv2d/Conv2D:output:00sequential/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
sequential/conv2d/BiasAdd?
-sequential/batch_normalization/ReadVariableOpReadVariableOp6sequential_batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential/batch_normalization/ReadVariableOp?
/sequential/batch_normalization/ReadVariableOp_1ReadVariableOp8sequential_batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype021
/sequential/batch_normalization/ReadVariableOp_1?
>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpGsequential_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02@
>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp?
@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIsequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02B
@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?
/sequential/batch_normalization/FusedBatchNormV3FusedBatchNormV3"sequential/conv2d/BiasAdd:output:05sequential/batch_normalization/ReadVariableOp:value:07sequential/batch_normalization/ReadVariableOp_1:value:0Fsequential/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Hsequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
is_training( 21
/sequential/batch_normalization/FusedBatchNormV3?
sequential/re_lu/ReluRelu3sequential/batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????2
sequential/re_lu/Relu?
)sequential/conv2d_1/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:	*
dtype02+
)sequential/conv2d_1/Conv2D/ReadVariableOp?
sequential/conv2d_1/Conv2DConv2D#sequential/re_lu/Relu:activations:01sequential/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????	*
paddingVALID*
strides
2
sequential/conv2d_1/Conv2D?
*sequential/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02,
*sequential/conv2d_1/BiasAdd/ReadVariableOp?
sequential/conv2d_1/BiasAddBiasAdd#sequential/conv2d_1/Conv2D:output:02sequential/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????	2
sequential/conv2d_1/BiasAddr
reshape/ShapeShape$sequential/conv2d_1/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape/Shape?
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack?
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1?
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2?
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slicet
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/2t
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :	2
reshape/Reshape/shape/3t
reshape/Reshape/shape/4Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/4t
reshape/Reshape/shape/5Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/5?
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0 reshape/Reshape/shape/4:output:0 reshape/Reshape/shape/5:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape?
reshape/ReshapeReshape$sequential/conv2d_1/BiasAdd:output:0reshape/Reshape/shape:output:0*
T0*7
_output_shapes%
#:!?????????	2
reshape/Reshape?
ExtractImagePatchesExtractImagePatchesx*
T0*/
_output_shapes
:?????????6*
ksizes
*
paddingSAME*
rates
*
strides
2
ExtractImagePatcheso
reshape_1/ShapeShapeExtractImagePatches:patches:0*
T0*
_output_shapes
:2
reshape_1/Shape?
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_1/strided_slice/stack?
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_1?
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_2?
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_1/strided_slicex
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/1x
reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/2x
reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :	2
reshape_1/Reshape/shape/3x
reshape_1/Reshape/shape/4Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/4x
reshape_1/Reshape/shape/5Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/5?
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0"reshape_1/Reshape/shape/2:output:0"reshape_1/Reshape/shape/3:output:0"reshape_1/Reshape/shape/4:output:0"reshape_1/Reshape/shape/5:output:0*
N*
T0*
_output_shapes
:2
reshape_1/Reshape/shape?
reshape_1/ReshapeReshapeExtractImagePatches:patches:0 reshape_1/Reshape/shape:output:0*
T0*7
_output_shapes%
#:!?????????	2
reshape_1/Reshape?
MulMulreshape/Reshape:output:0reshape_1/Reshape:output:0*
T0*7
_output_shapes%
#:!?????????	2
Mulp
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indicesx
SumSumMul:z:0Sum/reduction_indices:output:0*
T0*3
_output_shapes!
:?????????2
Sum^
reshape_2/ShapeShapeSum:output:0*
T0*
_output_shapes
:2
reshape_2/Shape?
reshape_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_2/strided_slice/stack?
reshape_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_2/strided_slice/stack_1?
reshape_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_2/strided_slice/stack_2?
reshape_2/strided_sliceStridedSlicereshape_2/Shape:output:0&reshape_2/strided_slice/stack:output:0(reshape_2/strided_slice/stack_1:output:0(reshape_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_2/strided_slicex
reshape_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_2/Reshape/shape/1x
reshape_2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_2/Reshape/shape/2x
reshape_2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_2/Reshape/shape/3?
reshape_2/Reshape/shapePack reshape_2/strided_slice:output:0"reshape_2/Reshape/shape/1:output:0"reshape_2/Reshape/shape/2:output:0"reshape_2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_2/Reshape/shape?
reshape_2/ReshapeReshapeSum:output:0 reshape_2/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????2
reshape_2/Reshape?

Identity_1Identityreshape_2/Reshape:output:0^NoOp*
T0*/
_output_shapes
:?????????2

Identity_1?
NoOpNoOp?^sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpA^sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1.^sequential/batch_normalization/ReadVariableOp0^sequential/batch_normalization/ReadVariableOp_1)^sequential/conv2d/BiasAdd/ReadVariableOp(^sequential/conv2d/Conv2D/ReadVariableOp+^sequential/conv2d_1/BiasAdd/ReadVariableOp*^sequential/conv2d_1/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????: : : : : : : : 2?
>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp2?
@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_12^
-sequential/batch_normalization/ReadVariableOp-sequential/batch_normalization/ReadVariableOp2b
/sequential/batch_normalization/ReadVariableOp_1/sequential/batch_normalization/ReadVariableOp_12T
(sequential/conv2d/BiasAdd/ReadVariableOp(sequential/conv2d/BiasAdd/ReadVariableOp2R
'sequential/conv2d/Conv2D/ReadVariableOp'sequential/conv2d/Conv2D/ReadVariableOp2X
*sequential/conv2d_1/BiasAdd/ReadVariableOp*sequential/conv2d_1/BiasAdd/ReadVariableOp2V
)sequential/conv2d_1/Conv2D/ReadVariableOp)sequential/conv2d_1/Conv2D/ReadVariableOp:R N
/
_output_shapes
:?????????

_user_specified_namex
?
?
&__inference_dense_layer_call_fn_306852

inputs
unknown:	?$@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_3040582
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????@2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????$: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????$
 
_user_specified_nameinputs
?
_
C__inference_re_lu_3_layer_call_and_return_conditional_losses_304037

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
e
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_306197

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
.__inference_InRFNet_Model_layer_call_fn_306000

inputs!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5:	
	unknown_6:	#
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:$

unknown_13:

unknown_14:$

unknown_15:

unknown_16:

unknown_17:

unknown_18:

unknown_19:

unknown_20:$

unknown_21:

unknown_22:$

unknown_23:

unknown_24:

unknown_25:

unknown_26:

unknown_27:

unknown_28:$

unknown_29:	

unknown_30:	

unknown_31:	?$@

unknown_32:@

unknown_33:@

unknown_34:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*>
_read_only_resource_inputs 
	
 !"#$*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_InRFNet_Model_layer_call_and_return_conditional_losses_3048452
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*x
_input_shapesg
e:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
$__inference_signature_wrapper_305262
input_1!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5:	
	unknown_6:	#
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:$

unknown_13:

unknown_14:$

unknown_15:

unknown_16:

unknown_17:

unknown_18:

unknown_19:

unknown_20:$

unknown_21:

unknown_22:$

unknown_23:

unknown_24:

unknown_25:

unknown_26:

unknown_27:

unknown_28:$

unknown_29:	

unknown_30:	

unknown_31:	?$@

unknown_32:@

unknown_33:@

unknown_34:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*F
_read_only_resource_inputs(
&$	
 !"#$*0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference__wrapped_model_3018472
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*x
_input_shapesg
e:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
?
N
2__inference_average_pooling2d_layer_call_fn_306986

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_average_pooling2d_layer_call_and_return_conditional_losses_3023022
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?

?
&__inference_inv_1_layer_call_fn_306161
x!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5:	
	unknown_6:	
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_inv_1_layer_call_and_return_conditional_losses_3037082
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:???????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
1
_output_shapes
:???????????

_user_specified_namex
?
?
O__inference_batch_normalization_layer_call_and_return_conditional_losses_307743

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????88:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????882

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????88: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????88
 
_user_specified_nameinputs
?
_
C__inference_re_lu_1_layer_call_and_return_conditional_losses_306581

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????882
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????882

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????88:W S
/
_output_shapes
:?????????88
 
_user_specified_nameinputs
?
e
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_306202

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????pp*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????pp2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_layer_call_and_return_conditional_losses_307725

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????88:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3w
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*/
_output_shapes
:?????????882

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????88: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:W S
/
_output_shapes
:?????????88
 
_user_specified_nameinputs
?
?
B__inference_conv2d_layer_call_and_return_conditional_losses_302904

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????88*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????882	
BiasAdds
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????882

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????88: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????88
 
_user_specified_nameinputs
?_
?
A__inference_inv_3_layer_call_and_return_conditional_losses_306699
xJ
0sequential_conv2d_conv2d_readvariableop_resource:?
1sequential_conv2d_biasadd_readvariableop_resource:D
6sequential_batch_normalization_readvariableop_resource:F
8sequential_batch_normalization_readvariableop_1_resource:U
Gsequential_batch_normalization_fusedbatchnormv3_readvariableop_resource:W
Isequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource:L
2sequential_conv2d_1_conv2d_readvariableop_resource:	A
3sequential_conv2d_1_biasadd_readvariableop_resource:	

identity_1??>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp?@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?-sequential/batch_normalization/ReadVariableOp?/sequential/batch_normalization/ReadVariableOp_1?(sequential/conv2d/BiasAdd/ReadVariableOp?'sequential/conv2d/Conv2D/ReadVariableOp?*sequential/conv2d_1/BiasAdd/ReadVariableOp?)sequential/conv2d_1/Conv2D/ReadVariableOp]
IdentityIdentityx*
T0*/
_output_shapes
:?????????2

Identity?
'sequential/conv2d/Conv2D/ReadVariableOpReadVariableOp0sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02)
'sequential/conv2d/Conv2D/ReadVariableOp?
sequential/conv2d/Conv2DConv2DIdentity:output:0/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
sequential/conv2d/Conv2D?
(sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOp1sequential_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(sequential/conv2d/BiasAdd/ReadVariableOp?
sequential/conv2d/BiasAddBiasAdd!sequential/conv2d/Conv2D:output:00sequential/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
sequential/conv2d/BiasAdd?
-sequential/batch_normalization/ReadVariableOpReadVariableOp6sequential_batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential/batch_normalization/ReadVariableOp?
/sequential/batch_normalization/ReadVariableOp_1ReadVariableOp8sequential_batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype021
/sequential/batch_normalization/ReadVariableOp_1?
>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpGsequential_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02@
>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp?
@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIsequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02B
@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?
/sequential/batch_normalization/FusedBatchNormV3FusedBatchNormV3"sequential/conv2d/BiasAdd:output:05sequential/batch_normalization/ReadVariableOp:value:07sequential/batch_normalization/ReadVariableOp_1:value:0Fsequential/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Hsequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????:::::*
epsilon%o?:*
is_training( 21
/sequential/batch_normalization/FusedBatchNormV3?
sequential/re_lu/ReluRelu3sequential/batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????2
sequential/re_lu/Relu?
)sequential/conv2d_1/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:	*
dtype02+
)sequential/conv2d_1/Conv2D/ReadVariableOp?
sequential/conv2d_1/Conv2DConv2D#sequential/re_lu/Relu:activations:01sequential/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????	*
paddingVALID*
strides
2
sequential/conv2d_1/Conv2D?
*sequential/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02,
*sequential/conv2d_1/BiasAdd/ReadVariableOp?
sequential/conv2d_1/BiasAddBiasAdd#sequential/conv2d_1/Conv2D:output:02sequential/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????	2
sequential/conv2d_1/BiasAddr
reshape/ShapeShape$sequential/conv2d_1/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape/Shape?
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack?
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1?
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2?
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slicet
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/2t
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :	2
reshape/Reshape/shape/3t
reshape/Reshape/shape/4Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/4t
reshape/Reshape/shape/5Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/5?
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0 reshape/Reshape/shape/4:output:0 reshape/Reshape/shape/5:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape?
reshape/ReshapeReshape$sequential/conv2d_1/BiasAdd:output:0reshape/Reshape/shape:output:0*
T0*7
_output_shapes%
#:!?????????	2
reshape/Reshape?
ExtractImagePatchesExtractImagePatchesx*
T0*/
_output_shapes
:?????????6*
ksizes
*
paddingSAME*
rates
*
strides
2
ExtractImagePatcheso
reshape_1/ShapeShapeExtractImagePatches:patches:0*
T0*
_output_shapes
:2
reshape_1/Shape?
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_1/strided_slice/stack?
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_1?
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_2?
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_1/strided_slicex
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/1x
reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/2x
reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :	2
reshape_1/Reshape/shape/3x
reshape_1/Reshape/shape/4Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/4x
reshape_1/Reshape/shape/5Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/5?
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0"reshape_1/Reshape/shape/2:output:0"reshape_1/Reshape/shape/3:output:0"reshape_1/Reshape/shape/4:output:0"reshape_1/Reshape/shape/5:output:0*
N*
T0*
_output_shapes
:2
reshape_1/Reshape/shape?
reshape_1/ReshapeReshapeExtractImagePatches:patches:0 reshape_1/Reshape/shape:output:0*
T0*7
_output_shapes%
#:!?????????	2
reshape_1/Reshape?
MulMulreshape/Reshape:output:0reshape_1/Reshape:output:0*
T0*7
_output_shapes%
#:!?????????	2
Mulp
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indicesx
SumSumMul:z:0Sum/reduction_indices:output:0*
T0*3
_output_shapes!
:?????????2
Sum^
reshape_2/ShapeShapeSum:output:0*
T0*
_output_shapes
:2
reshape_2/Shape?
reshape_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_2/strided_slice/stack?
reshape_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_2/strided_slice/stack_1?
reshape_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_2/strided_slice/stack_2?
reshape_2/strided_sliceStridedSlicereshape_2/Shape:output:0&reshape_2/strided_slice/stack:output:0(reshape_2/strided_slice/stack_1:output:0(reshape_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_2/strided_slicex
reshape_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_2/Reshape/shape/1x
reshape_2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_2/Reshape/shape/2x
reshape_2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_2/Reshape/shape/3?
reshape_2/Reshape/shapePack reshape_2/strided_slice:output:0"reshape_2/Reshape/shape/1:output:0"reshape_2/Reshape/shape/2:output:0"reshape_2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_2/Reshape/shape?
reshape_2/ReshapeReshapeSum:output:0 reshape_2/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????2
reshape_2/Reshape?

Identity_1Identityreshape_2/Reshape:output:0^NoOp*
T0*/
_output_shapes
:?????????2

Identity_1?
NoOpNoOp?^sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpA^sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1.^sequential/batch_normalization/ReadVariableOp0^sequential/batch_normalization/ReadVariableOp_1)^sequential/conv2d/BiasAdd/ReadVariableOp(^sequential/conv2d/Conv2D/ReadVariableOp+^sequential/conv2d_1/BiasAdd/ReadVariableOp*^sequential/conv2d_1/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????: : : : : : : : 2?
>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp2?
@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_12^
-sequential/batch_normalization/ReadVariableOp-sequential/batch_normalization/ReadVariableOp2b
/sequential/batch_normalization/ReadVariableOp_1/sequential/batch_normalization/ReadVariableOp_12T
(sequential/conv2d/BiasAdd/ReadVariableOp(sequential/conv2d/BiasAdd/ReadVariableOp2R
'sequential/conv2d/Conv2D/ReadVariableOp'sequential/conv2d/Conv2D/ReadVariableOp2X
*sequential/conv2d_1/BiasAdd/ReadVariableOp*sequential/conv2d_1/BiasAdd/ReadVariableOp2V
)sequential/conv2d_1/Conv2D/ReadVariableOp)sequential/conv2d_1/Conv2D/ReadVariableOp:R N
/
_output_shapes
:?????????

_user_specified_namex
?
?
O__inference_batch_normalization_layer_call_and_return_conditional_losses_302013

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3y
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:???????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?i
?	
B__inference_inv_22_layer_call_and_return_conditional_losses_304533
xJ
0sequential_conv2d_conv2d_readvariableop_resource:?
1sequential_conv2d_biasadd_readvariableop_resource:D
6sequential_batch_normalization_readvariableop_resource:F
8sequential_batch_normalization_readvariableop_1_resource:U
Gsequential_batch_normalization_fusedbatchnormv3_readvariableop_resource:W
Isequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource:L
2sequential_conv2d_1_conv2d_readvariableop_resource:A
3sequential_conv2d_1_biasadd_readvariableop_resource:
identity??-sequential/batch_normalization/AssignNewValue?/sequential/batch_normalization/AssignNewValue_1?>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp?@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?-sequential/batch_normalization/ReadVariableOp?/sequential/batch_normalization/ReadVariableOp_1?(sequential/conv2d/BiasAdd/ReadVariableOp?'sequential/conv2d/Conv2D/ReadVariableOp?*sequential/conv2d_1/BiasAdd/ReadVariableOp?)sequential/conv2d_1/Conv2D/ReadVariableOp?
average_pooling2d/AvgPoolAvgPoolx*
T0*/
_output_shapes
:?????????88*
ksize
*
paddingSAME*
strides
2
average_pooling2d/AvgPool?
'sequential/conv2d/Conv2D/ReadVariableOpReadVariableOp0sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02)
'sequential/conv2d/Conv2D/ReadVariableOp?
sequential/conv2d/Conv2DConv2D"average_pooling2d/AvgPool:output:0/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????88*
paddingVALID*
strides
2
sequential/conv2d/Conv2D?
(sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOp1sequential_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(sequential/conv2d/BiasAdd/ReadVariableOp?
sequential/conv2d/BiasAddBiasAdd!sequential/conv2d/Conv2D:output:00sequential/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????882
sequential/conv2d/BiasAdd?
-sequential/batch_normalization/ReadVariableOpReadVariableOp6sequential_batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential/batch_normalization/ReadVariableOp?
/sequential/batch_normalization/ReadVariableOp_1ReadVariableOp8sequential_batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype021
/sequential/batch_normalization/ReadVariableOp_1?
>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpGsequential_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02@
>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp?
@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIsequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02B
@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?
/sequential/batch_normalization/FusedBatchNormV3FusedBatchNormV3"sequential/conv2d/BiasAdd:output:05sequential/batch_normalization/ReadVariableOp:value:07sequential/batch_normalization/ReadVariableOp_1:value:0Fsequential/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Hsequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????88:::::*
epsilon%o?:*
exponential_avg_factor%
?#<21
/sequential/batch_normalization/FusedBatchNormV3?
-sequential/batch_normalization/AssignNewValueAssignVariableOpGsequential_batch_normalization_fusedbatchnormv3_readvariableop_resource<sequential/batch_normalization/FusedBatchNormV3:batch_mean:0?^sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02/
-sequential/batch_normalization/AssignNewValue?
/sequential/batch_normalization/AssignNewValue_1AssignVariableOpIsequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource@sequential/batch_normalization/FusedBatchNormV3:batch_variance:0A^sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype021
/sequential/batch_normalization/AssignNewValue_1?
sequential/re_lu/ReluRelu3sequential/batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????882
sequential/re_lu/Relu?
)sequential/conv2d_1/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02+
)sequential/conv2d_1/Conv2D/ReadVariableOp?
sequential/conv2d_1/Conv2DConv2D#sequential/re_lu/Relu:activations:01sequential/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????88*
paddingVALID*
strides
2
sequential/conv2d_1/Conv2D?
*sequential/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*sequential/conv2d_1/BiasAdd/ReadVariableOp?
sequential/conv2d_1/BiasAddBiasAdd#sequential/conv2d_1/Conv2D:output:02sequential/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????882
sequential/conv2d_1/BiasAddr
reshape/ShapeShape$sequential/conv2d_1/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape/Shape?
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack?
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1?
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2?
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slicet
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :82
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :82
reshape/Reshape/shape/2t
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/3t
reshape/Reshape/shape/4Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/4t
reshape/Reshape/shape/5Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/5?
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0 reshape/Reshape/shape/4:output:0 reshape/Reshape/shape/5:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape?
reshape/ReshapeReshape$sequential/conv2d_1/BiasAdd:output:0reshape/Reshape/shape:output:0*
T0*7
_output_shapes%
#:!?????????882
reshape/Reshape?
ExtractImagePatchesExtractImagePatchesx*
T0*/
_output_shapes
:?????????88	*
ksizes
*
paddingSAME*
rates
*
strides
2
ExtractImagePatcheso
reshape_1/ShapeShapeExtractImagePatches:patches:0*
T0*
_output_shapes
:2
reshape_1/Shape?
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_1/strided_slice/stack?
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_1?
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_2?
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_1/strided_slicex
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :82
reshape_1/Reshape/shape/1x
reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :82
reshape_1/Reshape/shape/2x
reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/3x
reshape_1/Reshape/shape/4Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/4x
reshape_1/Reshape/shape/5Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/5?
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0"reshape_1/Reshape/shape/2:output:0"reshape_1/Reshape/shape/3:output:0"reshape_1/Reshape/shape/4:output:0"reshape_1/Reshape/shape/5:output:0*
N*
T0*
_output_shapes
:2
reshape_1/Reshape/shape?
reshape_1/ReshapeReshapeExtractImagePatches:patches:0 reshape_1/Reshape/shape:output:0*
T0*7
_output_shapes%
#:!?????????882
reshape_1/Reshape?
MulMulreshape/Reshape:output:0reshape_1/Reshape:output:0*
T0*7
_output_shapes%
#:!?????????882
Mulp
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indicesx
SumSumMul:z:0Sum/reduction_indices:output:0*
T0*3
_output_shapes!
:?????????882
Sum^
reshape_2/ShapeShapeSum:output:0*
T0*
_output_shapes
:2
reshape_2/Shape?
reshape_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_2/strided_slice/stack?
reshape_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_2/strided_slice/stack_1?
reshape_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_2/strided_slice/stack_2?
reshape_2/strided_sliceStridedSlicereshape_2/Shape:output:0&reshape_2/strided_slice/stack:output:0(reshape_2/strided_slice/stack_1:output:0(reshape_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_2/strided_slicex
reshape_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :82
reshape_2/Reshape/shape/1x
reshape_2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :82
reshape_2/Reshape/shape/2x
reshape_2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_2/Reshape/shape/3?
reshape_2/Reshape/shapePack reshape_2/strided_slice:output:0"reshape_2/Reshape/shape/1:output:0"reshape_2/Reshape/shape/2:output:0"reshape_2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_2/Reshape/shape?
reshape_2/ReshapeReshapeSum:output:0 reshape_2/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????882
reshape_2/Reshape}
IdentityIdentityreshape_2/Reshape:output:0^NoOp*
T0*/
_output_shapes
:?????????882

Identity?
NoOpNoOp.^sequential/batch_normalization/AssignNewValue0^sequential/batch_normalization/AssignNewValue_1?^sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpA^sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1.^sequential/batch_normalization/ReadVariableOp0^sequential/batch_normalization/ReadVariableOp_1)^sequential/conv2d/BiasAdd/ReadVariableOp(^sequential/conv2d/Conv2D/ReadVariableOp+^sequential/conv2d_1/BiasAdd/ReadVariableOp*^sequential/conv2d_1/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????pp: : : : : : : : 2^
-sequential/batch_normalization/AssignNewValue-sequential/batch_normalization/AssignNewValue2b
/sequential/batch_normalization/AssignNewValue_1/sequential/batch_normalization/AssignNewValue_12?
>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp2?
@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_12^
-sequential/batch_normalization/ReadVariableOp-sequential/batch_normalization/ReadVariableOp2b
/sequential/batch_normalization/ReadVariableOp_1/sequential/batch_normalization/ReadVariableOp_12T
(sequential/conv2d/BiasAdd/ReadVariableOp(sequential/conv2d/BiasAdd/ReadVariableOp2R
'sequential/conv2d/Conv2D/ReadVariableOp'sequential/conv2d/Conv2D/ReadVariableOp2X
*sequential/conv2d_1/BiasAdd/ReadVariableOp*sequential/conv2d_1/BiasAdd/ReadVariableOp2V
)sequential/conv2d_1/Conv2D/ReadVariableOp)sequential/conv2d_1/Conv2D/ReadVariableOp:R N
/
_output_shapes
:?????????pp

_user_specified_namex
?
L
0__inference_max_pooling2d_1_layer_call_fn_306624

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_3031942
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
??
?A
"__inference__traced_restore_308635
file_prefix0
assignvariableop_dense_kernel:	?$@+
assignvariableop_1_dense_bias:@3
!assignvariableop_2_dense_1_kernel:@-
assignvariableop_3_dense_1_bias:&
assignvariableop_4_adam_iter:	 (
assignvariableop_5_adam_beta_1: (
assignvariableop_6_adam_beta_2: '
assignvariableop_7_adam_decay: /
%assignvariableop_8_adam_learning_rate: :
 assignvariableop_9_conv2d_kernel:-
assignvariableop_10_conv2d_bias:;
-assignvariableop_11_batch_normalization_gamma::
,assignvariableop_12_batch_normalization_beta:=
#assignvariableop_13_conv2d_1_kernel:	/
!assignvariableop_14_conv2d_1_bias:	=
#assignvariableop_15_conv2d_kernel_1:/
!assignvariableop_16_conv2d_bias_1:=
/assignvariableop_17_batch_normalization_gamma_1:<
.assignvariableop_18_batch_normalization_beta_1:?
%assignvariableop_19_conv2d_1_kernel_1:1
#assignvariableop_20_conv2d_1_bias_1:=
#assignvariableop_21_conv2d_kernel_2:/
!assignvariableop_22_conv2d_bias_2:=
/assignvariableop_23_batch_normalization_gamma_2:<
.assignvariableop_24_batch_normalization_beta_2:?
%assignvariableop_25_conv2d_1_kernel_2:1
#assignvariableop_26_conv2d_1_bias_2:=
#assignvariableop_27_conv2d_kernel_3:/
!assignvariableop_28_conv2d_bias_3:=
/assignvariableop_29_batch_normalization_gamma_3:<
.assignvariableop_30_batch_normalization_beta_3:?
%assignvariableop_31_conv2d_1_kernel_3:	1
#assignvariableop_32_conv2d_1_bias_3:	A
3assignvariableop_33_batch_normalization_moving_mean:E
7assignvariableop_34_batch_normalization_moving_variance:C
5assignvariableop_35_batch_normalization_moving_mean_1:G
9assignvariableop_36_batch_normalization_moving_variance_1:C
5assignvariableop_37_batch_normalization_moving_mean_2:G
9assignvariableop_38_batch_normalization_moving_variance_2:C
5assignvariableop_39_batch_normalization_moving_mean_3:G
9assignvariableop_40_batch_normalization_moving_variance_3:#
assignvariableop_41_total: #
assignvariableop_42_count: %
assignvariableop_43_total_1: %
assignvariableop_44_count_1: :
'assignvariableop_45_adam_dense_kernel_m:	?$@3
%assignvariableop_46_adam_dense_bias_m:@;
)assignvariableop_47_adam_dense_1_kernel_m:@5
'assignvariableop_48_adam_dense_1_bias_m:B
(assignvariableop_49_adam_conv2d_kernel_m:4
&assignvariableop_50_adam_conv2d_bias_m:B
4assignvariableop_51_adam_batch_normalization_gamma_m:A
3assignvariableop_52_adam_batch_normalization_beta_m:D
*assignvariableop_53_adam_conv2d_1_kernel_m:	6
(assignvariableop_54_adam_conv2d_1_bias_m:	D
*assignvariableop_55_adam_conv2d_kernel_m_1:6
(assignvariableop_56_adam_conv2d_bias_m_1:D
6assignvariableop_57_adam_batch_normalization_gamma_m_1:C
5assignvariableop_58_adam_batch_normalization_beta_m_1:F
,assignvariableop_59_adam_conv2d_1_kernel_m_1:8
*assignvariableop_60_adam_conv2d_1_bias_m_1:D
*assignvariableop_61_adam_conv2d_kernel_m_2:6
(assignvariableop_62_adam_conv2d_bias_m_2:D
6assignvariableop_63_adam_batch_normalization_gamma_m_2:C
5assignvariableop_64_adam_batch_normalization_beta_m_2:F
,assignvariableop_65_adam_conv2d_1_kernel_m_2:8
*assignvariableop_66_adam_conv2d_1_bias_m_2:D
*assignvariableop_67_adam_conv2d_kernel_m_3:6
(assignvariableop_68_adam_conv2d_bias_m_3:D
6assignvariableop_69_adam_batch_normalization_gamma_m_3:C
5assignvariableop_70_adam_batch_normalization_beta_m_3:F
,assignvariableop_71_adam_conv2d_1_kernel_m_3:	8
*assignvariableop_72_adam_conv2d_1_bias_m_3:	:
'assignvariableop_73_adam_dense_kernel_v:	?$@3
%assignvariableop_74_adam_dense_bias_v:@;
)assignvariableop_75_adam_dense_1_kernel_v:@5
'assignvariableop_76_adam_dense_1_bias_v:B
(assignvariableop_77_adam_conv2d_kernel_v:4
&assignvariableop_78_adam_conv2d_bias_v:B
4assignvariableop_79_adam_batch_normalization_gamma_v:A
3assignvariableop_80_adam_batch_normalization_beta_v:D
*assignvariableop_81_adam_conv2d_1_kernel_v:	6
(assignvariableop_82_adam_conv2d_1_bias_v:	D
*assignvariableop_83_adam_conv2d_kernel_v_1:6
(assignvariableop_84_adam_conv2d_bias_v_1:D
6assignvariableop_85_adam_batch_normalization_gamma_v_1:C
5assignvariableop_86_adam_batch_normalization_beta_v_1:F
,assignvariableop_87_adam_conv2d_1_kernel_v_1:8
*assignvariableop_88_adam_conv2d_1_bias_v_1:D
*assignvariableop_89_adam_conv2d_kernel_v_2:6
(assignvariableop_90_adam_conv2d_bias_v_2:D
6assignvariableop_91_adam_batch_normalization_gamma_v_2:C
5assignvariableop_92_adam_batch_normalization_beta_v_2:F
,assignvariableop_93_adam_conv2d_1_kernel_v_2:8
*assignvariableop_94_adam_conv2d_1_bias_v_2:D
*assignvariableop_95_adam_conv2d_kernel_v_3:6
(assignvariableop_96_adam_conv2d_bias_v_3:D
6assignvariableop_97_adam_batch_normalization_gamma_v_3:C
5assignvariableop_98_adam_batch_normalization_beta_v_3:F
,assignvariableop_99_adam_conv2d_1_kernel_v_3:	9
+assignvariableop_100_adam_conv2d_1_bias_v_3:	
identity_102??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_100?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_67?AssignVariableOp_68?AssignVariableOp_69?AssignVariableOp_7?AssignVariableOp_70?AssignVariableOp_71?AssignVariableOp_72?AssignVariableOp_73?AssignVariableOp_74?AssignVariableOp_75?AssignVariableOp_76?AssignVariableOp_77?AssignVariableOp_78?AssignVariableOp_79?AssignVariableOp_8?AssignVariableOp_80?AssignVariableOp_81?AssignVariableOp_82?AssignVariableOp_83?AssignVariableOp_84?AssignVariableOp_85?AssignVariableOp_86?AssignVariableOp_87?AssignVariableOp_88?AssignVariableOp_89?AssignVariableOp_9?AssignVariableOp_90?AssignVariableOp_91?AssignVariableOp_92?AssignVariableOp_93?AssignVariableOp_94?AssignVariableOp_95?AssignVariableOp_96?AssignVariableOp_97?AssignVariableOp_98?AssignVariableOp_99?5
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:f*
dtype0*?4
value?4B?4fB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/19/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/20/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/21/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/22/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/23/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:f*
dtype0*?
value?B?fB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*t
dtypesj
h2f	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_dense_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_iterIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_beta_1Identity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_beta_2Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_decayIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp%assignvariableop_8_adam_learning_rateIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp assignvariableop_9_conv2d_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOpassignvariableop_10_conv2d_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp-assignvariableop_11_batch_normalization_gammaIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp,assignvariableop_12_batch_normalization_betaIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp#assignvariableop_13_conv2d_1_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp!assignvariableop_14_conv2d_1_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp#assignvariableop_15_conv2d_kernel_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp!assignvariableop_16_conv2d_bias_1Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp/assignvariableop_17_batch_normalization_gamma_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp.assignvariableop_18_batch_normalization_beta_1Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp%assignvariableop_19_conv2d_1_kernel_1Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp#assignvariableop_20_conv2d_1_bias_1Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp#assignvariableop_21_conv2d_kernel_2Identity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp!assignvariableop_22_conv2d_bias_2Identity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp/assignvariableop_23_batch_normalization_gamma_2Identity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp.assignvariableop_24_batch_normalization_beta_2Identity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp%assignvariableop_25_conv2d_1_kernel_2Identity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp#assignvariableop_26_conv2d_1_bias_2Identity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp#assignvariableop_27_conv2d_kernel_3Identity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp!assignvariableop_28_conv2d_bias_3Identity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp/assignvariableop_29_batch_normalization_gamma_3Identity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp.assignvariableop_30_batch_normalization_beta_3Identity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp%assignvariableop_31_conv2d_1_kernel_3Identity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp#assignvariableop_32_conv2d_1_bias_3Identity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp3assignvariableop_33_batch_normalization_moving_meanIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp7assignvariableop_34_batch_normalization_moving_varianceIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp5assignvariableop_35_batch_normalization_moving_mean_1Identity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp9assignvariableop_36_batch_normalization_moving_variance_1Identity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp5assignvariableop_37_batch_normalization_moving_mean_2Identity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp9assignvariableop_38_batch_normalization_moving_variance_2Identity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp5assignvariableop_39_batch_normalization_moving_mean_3Identity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp9assignvariableop_40_batch_normalization_moving_variance_3Identity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOpassignvariableop_41_totalIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOpassignvariableop_42_countIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOpassignvariableop_43_total_1Identity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOpassignvariableop_44_count_1Identity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp'assignvariableop_45_adam_dense_kernel_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp%assignvariableop_46_adam_dense_bias_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp)assignvariableop_47_adam_dense_1_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp'assignvariableop_48_adam_dense_1_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOp(assignvariableop_49_adam_conv2d_kernel_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOp&assignvariableop_50_adam_conv2d_bias_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOp4assignvariableop_51_adam_batch_normalization_gamma_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOp3assignvariableop_52_adam_batch_normalization_beta_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOp*assignvariableop_53_adam_conv2d_1_kernel_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOp(assignvariableop_54_adam_conv2d_1_bias_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55?
AssignVariableOp_55AssignVariableOp*assignvariableop_55_adam_conv2d_kernel_m_1Identity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56?
AssignVariableOp_56AssignVariableOp(assignvariableop_56_adam_conv2d_bias_m_1Identity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57?
AssignVariableOp_57AssignVariableOp6assignvariableop_57_adam_batch_normalization_gamma_m_1Identity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58?
AssignVariableOp_58AssignVariableOp5assignvariableop_58_adam_batch_normalization_beta_m_1Identity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59?
AssignVariableOp_59AssignVariableOp,assignvariableop_59_adam_conv2d_1_kernel_m_1Identity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60?
AssignVariableOp_60AssignVariableOp*assignvariableop_60_adam_conv2d_1_bias_m_1Identity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61?
AssignVariableOp_61AssignVariableOp*assignvariableop_61_adam_conv2d_kernel_m_2Identity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62?
AssignVariableOp_62AssignVariableOp(assignvariableop_62_adam_conv2d_bias_m_2Identity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63?
AssignVariableOp_63AssignVariableOp6assignvariableop_63_adam_batch_normalization_gamma_m_2Identity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64?
AssignVariableOp_64AssignVariableOp5assignvariableop_64_adam_batch_normalization_beta_m_2Identity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65?
AssignVariableOp_65AssignVariableOp,assignvariableop_65_adam_conv2d_1_kernel_m_2Identity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66?
AssignVariableOp_66AssignVariableOp*assignvariableop_66_adam_conv2d_1_bias_m_2Identity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67?
AssignVariableOp_67AssignVariableOp*assignvariableop_67_adam_conv2d_kernel_m_3Identity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68?
AssignVariableOp_68AssignVariableOp(assignvariableop_68_adam_conv2d_bias_m_3Identity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69?
AssignVariableOp_69AssignVariableOp6assignvariableop_69_adam_batch_normalization_gamma_m_3Identity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70?
AssignVariableOp_70AssignVariableOp5assignvariableop_70_adam_batch_normalization_beta_m_3Identity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71?
AssignVariableOp_71AssignVariableOp,assignvariableop_71_adam_conv2d_1_kernel_m_3Identity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72?
AssignVariableOp_72AssignVariableOp*assignvariableop_72_adam_conv2d_1_bias_m_3Identity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73?
AssignVariableOp_73AssignVariableOp'assignvariableop_73_adam_dense_kernel_vIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74?
AssignVariableOp_74AssignVariableOp%assignvariableop_74_adam_dense_bias_vIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75?
AssignVariableOp_75AssignVariableOp)assignvariableop_75_adam_dense_1_kernel_vIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76?
AssignVariableOp_76AssignVariableOp'assignvariableop_76_adam_dense_1_bias_vIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_76n
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:2
Identity_77?
AssignVariableOp_77AssignVariableOp(assignvariableop_77_adam_conv2d_kernel_vIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_77n
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:2
Identity_78?
AssignVariableOp_78AssignVariableOp&assignvariableop_78_adam_conv2d_bias_vIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_78n
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:2
Identity_79?
AssignVariableOp_79AssignVariableOp4assignvariableop_79_adam_batch_normalization_gamma_vIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79n
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:2
Identity_80?
AssignVariableOp_80AssignVariableOp3assignvariableop_80_adam_batch_normalization_beta_vIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_80n
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:2
Identity_81?
AssignVariableOp_81AssignVariableOp*assignvariableop_81_adam_conv2d_1_kernel_vIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_81n
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:2
Identity_82?
AssignVariableOp_82AssignVariableOp(assignvariableop_82_adam_conv2d_1_bias_vIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_82n
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:2
Identity_83?
AssignVariableOp_83AssignVariableOp*assignvariableop_83_adam_conv2d_kernel_v_1Identity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_83n
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:2
Identity_84?
AssignVariableOp_84AssignVariableOp(assignvariableop_84_adam_conv2d_bias_v_1Identity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_84n
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:2
Identity_85?
AssignVariableOp_85AssignVariableOp6assignvariableop_85_adam_batch_normalization_gamma_v_1Identity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_85n
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:2
Identity_86?
AssignVariableOp_86AssignVariableOp5assignvariableop_86_adam_batch_normalization_beta_v_1Identity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_86n
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:2
Identity_87?
AssignVariableOp_87AssignVariableOp,assignvariableop_87_adam_conv2d_1_kernel_v_1Identity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_87n
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:2
Identity_88?
AssignVariableOp_88AssignVariableOp*assignvariableop_88_adam_conv2d_1_bias_v_1Identity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_88n
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:2
Identity_89?
AssignVariableOp_89AssignVariableOp*assignvariableop_89_adam_conv2d_kernel_v_2Identity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_89n
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:2
Identity_90?
AssignVariableOp_90AssignVariableOp(assignvariableop_90_adam_conv2d_bias_v_2Identity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_90n
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:2
Identity_91?
AssignVariableOp_91AssignVariableOp6assignvariableop_91_adam_batch_normalization_gamma_v_2Identity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_91n
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:2
Identity_92?
AssignVariableOp_92AssignVariableOp5assignvariableop_92_adam_batch_normalization_beta_v_2Identity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_92n
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:2
Identity_93?
AssignVariableOp_93AssignVariableOp,assignvariableop_93_adam_conv2d_1_kernel_v_2Identity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_93n
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:2
Identity_94?
AssignVariableOp_94AssignVariableOp*assignvariableop_94_adam_conv2d_1_bias_v_2Identity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_94n
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:2
Identity_95?
AssignVariableOp_95AssignVariableOp*assignvariableop_95_adam_conv2d_kernel_v_3Identity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_95n
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:2
Identity_96?
AssignVariableOp_96AssignVariableOp(assignvariableop_96_adam_conv2d_bias_v_3Identity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_96n
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:2
Identity_97?
AssignVariableOp_97AssignVariableOp6assignvariableop_97_adam_batch_normalization_gamma_v_3Identity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_97n
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:2
Identity_98?
AssignVariableOp_98AssignVariableOp5assignvariableop_98_adam_batch_normalization_beta_v_3Identity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_98n
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:2
Identity_99?
AssignVariableOp_99AssignVariableOp,assignvariableop_99_adam_conv2d_1_kernel_v_3Identity_99:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_99q
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:2
Identity_100?
AssignVariableOp_100AssignVariableOp+assignvariableop_100_adam_conv2d_1_bias_v_3Identity_100:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1009
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_101Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_101i
Identity_102IdentityIdentity_101:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_102?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"%
identity_102Identity_102:output:0*?
_input_shapes?
?: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102,
AssignVariableOp_100AssignVariableOp_1002*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_89AssignVariableOp_892(
AssignVariableOp_9AssignVariableOp_92*
AssignVariableOp_90AssignVariableOp_902*
AssignVariableOp_91AssignVariableOp_912*
AssignVariableOp_92AssignVariableOp_922*
AssignVariableOp_93AssignVariableOp_932*
AssignVariableOp_94AssignVariableOp_942*
AssignVariableOp_95AssignVariableOp_952*
AssignVariableOp_96AssignVariableOp_962*
AssignVariableOp_97AssignVariableOp_972*
AssignVariableOp_98AssignVariableOp_982*
AssignVariableOp_99AssignVariableOp_99:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?

?
+__inference_sequential_layer_call_fn_302980
conv2d_input!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????88**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_3029612
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????882

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????88: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
/
_output_shapes
:?????????88
&
_user_specified_nameconv2d_input
?
?
D__inference_conv2d_1_layer_call_and_return_conditional_losses_307815

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????88*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????882	
BiasAdds
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????882

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????88: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????88
 
_user_specified_nameinputs
?
_
C__inference_flatten_layer_call_and_return_conditional_losses_306827

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????`  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????$2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????$2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
4__inference_batch_normalization_layer_call_fn_307451

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_3021152
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:???????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
B__inference_conv2d_layer_call_and_return_conditional_losses_307662

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????88*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????882	
BiasAdds
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????882

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????88: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????88
 
_user_specified_nameinputs
?
?
F__inference_sequential_layer_call_and_return_conditional_losses_302183

inputs'
conv2d_302162:
conv2d_302164:(
batch_normalization_302167:(
batch_normalization_302169:(
batch_normalization_302171:(
batch_normalization_302173:)
conv2d_1_302177:	
conv2d_1_302179:	
identity??+batch_normalization/StatefulPartitionedCall?conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_302162conv2d_302164*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_3019902 
conv2d/StatefulPartitionedCall?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_302167batch_normalization_302169batch_normalization_302171batch_normalization_302173*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_3021152-
+batch_normalization/StatefulPartitionedCall?
re_lu/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_re_lu_layer_call_and_return_conditional_losses_3020282
re_lu/PartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallre_lu/PartitionedCall:output:0conv2d_1_302177conv2d_1_302179*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_3020402"
 conv2d_1/StatefulPartitionedCall?
IdentityIdentity)conv2d_1/StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????	2

Identity?
NoOpNoOp,^batch_normalization/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:???????????: : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?/
?
F__inference_sequential_layer_call_and_return_conditional_losses_306934

inputs?
%conv2d_conv2d_readvariableop_resource:4
&conv2d_biasadd_readvariableop_resource:9
+batch_normalization_readvariableop_resource:;
-batch_normalization_readvariableop_1_resource:J
<batch_normalization_fusedbatchnormv3_readvariableop_resource:L
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource:A
'conv2d_1_conv2d_readvariableop_resource:	6
(conv2d_1_biasadd_readvariableop_resource:	
identity??"batch_normalization/AssignNewValue?$batch_normalization/AssignNewValue_1?3batch_normalization/FusedBatchNormV3/ReadVariableOp?5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?"batch_normalization/ReadVariableOp?$batch_normalization/ReadVariableOp_1?conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
conv2d/BiasAdd?
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype02$
"batch_normalization/ReadVariableOp?
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype02&
$batch_normalization/ReadVariableOp_1?
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOp?
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d/BiasAdd:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2&
$batch_normalization/FusedBatchNormV3?
"batch_normalization/AssignNewValueAssignVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource1batch_normalization/FusedBatchNormV3:batch_mean:04^batch_normalization/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02$
"batch_normalization/AssignNewValue?
$batch_normalization/AssignNewValue_1AssignVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource5batch_normalization/FusedBatchNormV3:batch_variance:06^batch_normalization/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02&
$batch_normalization/AssignNewValue_1?

re_lu/ReluRelu(batch_normalization/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????2

re_lu/Relu?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:	*
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2Dre_lu/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????	*
paddingVALID*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????	2
conv2d_1/BiasAdd~
IdentityIdentityconv2d_1/BiasAdd:output:0^NoOp*
T0*1
_output_shapes
:???????????	2

Identity?
NoOpNoOp#^batch_normalization/AssignNewValue%^batch_normalization/AssignNewValue_14^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_1^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:???????????: : : : : : : : 2H
"batch_normalization/AssignNewValue"batch_normalization/AssignNewValue2L
$batch_normalization/AssignNewValue_1$batch_normalization/AssignNewValue_12j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?	
?
4__inference_batch_normalization_layer_call_fn_307756

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_3027832
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?

?
+__inference_sequential_layer_call_fn_307287

inputs!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5:	
	unknown_6:	
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????	**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_3034072
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????	2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
]
A__inference_re_lu_layer_call_and_return_conditional_losses_303731

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:???????????2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
C__inference_dense_1_layer_call_and_return_conditional_losses_304075

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Softmaxl
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
.__inference_InRFNet_Model_layer_call_fn_304997
input_1!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5:	
	unknown_6:	#
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:$

unknown_13:

unknown_14:$

unknown_15:

unknown_16:

unknown_17:

unknown_18:

unknown_19:

unknown_20:$

unknown_21:

unknown_22:$

unknown_23:

unknown_24:

unknown_25:

unknown_26:

unknown_27:

unknown_28:$

unknown_29:	

unknown_30:	

unknown_31:	?$@

unknown_32:@

unknown_33:@

unknown_34:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*>
_read_only_resource_inputs 
	
 !"#$*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_InRFNet_Model_layer_call_and_return_conditional_losses_3048452
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*x
_input_shapesg
e:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
?
?
D__inference_conv2d_1_layer_call_and_return_conditional_losses_307987

inputs8
conv2d_readvariableop_resource:	-
biasadd_readvariableop_resource:	
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????	*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????	2	
BiasAdds
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????	2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_layer_call_and_return_conditional_losses_303273

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
D
(__inference_re_lu_2_layer_call_fn_306596

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????88* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_re_lu_2_layer_call_and_return_conditional_losses_3039272
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????882

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????88:W S
/
_output_shapes
:?????????88
 
_user_specified_nameinputs
?(
?
F__inference_sequential_layer_call_and_return_conditional_losses_307131

inputs?
%conv2d_conv2d_readvariableop_resource:4
&conv2d_biasadd_readvariableop_resource:9
+batch_normalization_readvariableop_resource:;
-batch_normalization_readvariableop_1_resource:J
<batch_normalization_fusedbatchnormv3_readvariableop_resource:L
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource:A
'conv2d_1_conv2d_readvariableop_resource:6
(conv2d_1_biasadd_readvariableop_resource:
identity??3batch_normalization/FusedBatchNormV3/ReadVariableOp?5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?"batch_normalization/ReadVariableOp?$batch_normalization/ReadVariableOp_1?conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????88*
paddingVALID*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????882
conv2d/BiasAdd?
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype02$
"batch_normalization/ReadVariableOp?
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype02&
$batch_normalization/ReadVariableOp_1?
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOp?
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d/BiasAdd:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????88:::::*
epsilon%o?:*
is_training( 2&
$batch_normalization/FusedBatchNormV3?

re_lu/ReluRelu(batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????882

re_lu/Relu?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2Dre_lu/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????88*
paddingVALID*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????882
conv2d_1/BiasAdd|
IdentityIdentityconv2d_1/BiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????882

Identity?
NoOpNoOp4^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_1^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????88: : : : : : : : 2j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????88
 
_user_specified_nameinputs
?
?
F__inference_sequential_layer_call_and_return_conditional_losses_302515

inputs'
conv2d_302459:
conv2d_302461:(
batch_normalization_302482:(
batch_normalization_302484:(
batch_normalization_302486:(
batch_normalization_302488:)
conv2d_1_302509:
conv2d_1_302511:
identity??+batch_normalization/StatefulPartitionedCall?conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_302459conv2d_302461*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????88*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_3024582 
conv2d/StatefulPartitionedCall?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_302482batch_normalization_302484batch_normalization_302486batch_normalization_302488*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????88*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_3024812-
+batch_normalization/StatefulPartitionedCall?
re_lu/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????88* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_re_lu_layer_call_and_return_conditional_losses_3024962
re_lu/PartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallre_lu/PartitionedCall:output:0conv2d_1_302509conv2d_1_302511*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????88*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_3025082"
 conv2d_1/StatefulPartitionedCall?
IdentityIdentity)conv2d_1/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????882

Identity?
NoOpNoOp,^batch_normalization/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????88: : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall:W S
/
_output_shapes
:?????????88
 
_user_specified_nameinputs
?
?
4__inference_batch_normalization_layer_call_fn_307438

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_3020132
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:???????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
B
&__inference_re_lu_layer_call_fn_307633

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????88* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_re_lu_layer_call_and_return_conditional_losses_3024962
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????882

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????88:W S
/
_output_shapes
:?????????88
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_layer_call_and_return_conditional_losses_302827

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
e
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_303737

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????pp*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????pp2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?`
?
B__inference_inv_21_layer_call_and_return_conditional_losses_306282
xJ
0sequential_conv2d_conv2d_readvariableop_resource:?
1sequential_conv2d_biasadd_readvariableop_resource:D
6sequential_batch_normalization_readvariableop_resource:F
8sequential_batch_normalization_readvariableop_1_resource:U
Gsequential_batch_normalization_fusedbatchnormv3_readvariableop_resource:W
Isequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource:L
2sequential_conv2d_1_conv2d_readvariableop_resource:A
3sequential_conv2d_1_biasadd_readvariableop_resource:
identity??>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp?@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?-sequential/batch_normalization/ReadVariableOp?/sequential/batch_normalization/ReadVariableOp_1?(sequential/conv2d/BiasAdd/ReadVariableOp?'sequential/conv2d/Conv2D/ReadVariableOp?*sequential/conv2d_1/BiasAdd/ReadVariableOp?)sequential/conv2d_1/Conv2D/ReadVariableOp?
average_pooling2d/AvgPoolAvgPoolx*
T0*/
_output_shapes
:?????????88*
ksize
*
paddingSAME*
strides
2
average_pooling2d/AvgPool?
'sequential/conv2d/Conv2D/ReadVariableOpReadVariableOp0sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02)
'sequential/conv2d/Conv2D/ReadVariableOp?
sequential/conv2d/Conv2DConv2D"average_pooling2d/AvgPool:output:0/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????88*
paddingVALID*
strides
2
sequential/conv2d/Conv2D?
(sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOp1sequential_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(sequential/conv2d/BiasAdd/ReadVariableOp?
sequential/conv2d/BiasAddBiasAdd!sequential/conv2d/Conv2D:output:00sequential/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????882
sequential/conv2d/BiasAdd?
-sequential/batch_normalization/ReadVariableOpReadVariableOp6sequential_batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential/batch_normalization/ReadVariableOp?
/sequential/batch_normalization/ReadVariableOp_1ReadVariableOp8sequential_batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype021
/sequential/batch_normalization/ReadVariableOp_1?
>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpGsequential_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02@
>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp?
@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIsequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02B
@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?
/sequential/batch_normalization/FusedBatchNormV3FusedBatchNormV3"sequential/conv2d/BiasAdd:output:05sequential/batch_normalization/ReadVariableOp:value:07sequential/batch_normalization/ReadVariableOp_1:value:0Fsequential/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Hsequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????88:::::*
epsilon%o?:*
is_training( 21
/sequential/batch_normalization/FusedBatchNormV3?
sequential/re_lu/ReluRelu3sequential/batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????882
sequential/re_lu/Relu?
)sequential/conv2d_1/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02+
)sequential/conv2d_1/Conv2D/ReadVariableOp?
sequential/conv2d_1/Conv2DConv2D#sequential/re_lu/Relu:activations:01sequential/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????88*
paddingVALID*
strides
2
sequential/conv2d_1/Conv2D?
*sequential/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*sequential/conv2d_1/BiasAdd/ReadVariableOp?
sequential/conv2d_1/BiasAddBiasAdd#sequential/conv2d_1/Conv2D:output:02sequential/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????882
sequential/conv2d_1/BiasAddr
reshape/ShapeShape$sequential/conv2d_1/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape/Shape?
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack?
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1?
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2?
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slicet
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :82
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :82
reshape/Reshape/shape/2t
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/3t
reshape/Reshape/shape/4Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/4t
reshape/Reshape/shape/5Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/5?
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0 reshape/Reshape/shape/4:output:0 reshape/Reshape/shape/5:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape?
reshape/ReshapeReshape$sequential/conv2d_1/BiasAdd:output:0reshape/Reshape/shape:output:0*
T0*7
_output_shapes%
#:!?????????882
reshape/Reshape?
ExtractImagePatchesExtractImagePatchesx*
T0*/
_output_shapes
:?????????88	*
ksizes
*
paddingSAME*
rates
*
strides
2
ExtractImagePatcheso
reshape_1/ShapeShapeExtractImagePatches:patches:0*
T0*
_output_shapes
:2
reshape_1/Shape?
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_1/strided_slice/stack?
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_1?
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_2?
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_1/strided_slicex
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :82
reshape_1/Reshape/shape/1x
reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :82
reshape_1/Reshape/shape/2x
reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/3x
reshape_1/Reshape/shape/4Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/4x
reshape_1/Reshape/shape/5Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/5?
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0"reshape_1/Reshape/shape/2:output:0"reshape_1/Reshape/shape/3:output:0"reshape_1/Reshape/shape/4:output:0"reshape_1/Reshape/shape/5:output:0*
N*
T0*
_output_shapes
:2
reshape_1/Reshape/shape?
reshape_1/ReshapeReshapeExtractImagePatches:patches:0 reshape_1/Reshape/shape:output:0*
T0*7
_output_shapes%
#:!?????????882
reshape_1/Reshape?
MulMulreshape/Reshape:output:0reshape_1/Reshape:output:0*
T0*7
_output_shapes%
#:!?????????882
Mulp
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indicesx
SumSumMul:z:0Sum/reduction_indices:output:0*
T0*3
_output_shapes!
:?????????882
Sum^
reshape_2/ShapeShapeSum:output:0*
T0*
_output_shapes
:2
reshape_2/Shape?
reshape_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_2/strided_slice/stack?
reshape_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_2/strided_slice/stack_1?
reshape_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_2/strided_slice/stack_2?
reshape_2/strided_sliceStridedSlicereshape_2/Shape:output:0&reshape_2/strided_slice/stack:output:0(reshape_2/strided_slice/stack_1:output:0(reshape_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_2/strided_slicex
reshape_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :82
reshape_2/Reshape/shape/1x
reshape_2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :82
reshape_2/Reshape/shape/2x
reshape_2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_2/Reshape/shape/3?
reshape_2/Reshape/shapePack reshape_2/strided_slice:output:0"reshape_2/Reshape/shape/1:output:0"reshape_2/Reshape/shape/2:output:0"reshape_2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_2/Reshape/shape?
reshape_2/ReshapeReshapeSum:output:0 reshape_2/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????882
reshape_2/Reshape}
IdentityIdentityreshape_2/Reshape:output:0^NoOp*
T0*/
_output_shapes
:?????????882

Identity?
NoOpNoOp?^sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpA^sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1.^sequential/batch_normalization/ReadVariableOp0^sequential/batch_normalization/ReadVariableOp_1)^sequential/conv2d/BiasAdd/ReadVariableOp(^sequential/conv2d/Conv2D/ReadVariableOp+^sequential/conv2d_1/BiasAdd/ReadVariableOp*^sequential/conv2d_1/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????pp: : : : : : : : 2?
>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp2?
@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_12^
-sequential/batch_normalization/ReadVariableOp-sequential/batch_normalization/ReadVariableOp2b
/sequential/batch_normalization/ReadVariableOp_1/sequential/batch_normalization/ReadVariableOp_12T
(sequential/conv2d/BiasAdd/ReadVariableOp(sequential/conv2d/BiasAdd/ReadVariableOp2R
'sequential/conv2d/Conv2D/ReadVariableOp'sequential/conv2d/Conv2D/ReadVariableOp2X
*sequential/conv2d_1/BiasAdd/ReadVariableOp*sequential/conv2d_1/BiasAdd/ReadVariableOp2V
)sequential/conv2d_1/Conv2D/ReadVariableOp)sequential/conv2d_1/Conv2D/ReadVariableOp:R N
/
_output_shapes
:?????????pp

_user_specified_namex
?
?
)__inference_conv2d_1_layer_call_fn_307996

inputs!
unknown:	
	unknown_0:	
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_3034002
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????	2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?`
?
A__inference_inv_1_layer_call_and_return_conditional_losses_303708
xJ
0sequential_conv2d_conv2d_readvariableop_resource:?
1sequential_conv2d_biasadd_readvariableop_resource:D
6sequential_batch_normalization_readvariableop_resource:F
8sequential_batch_normalization_readvariableop_1_resource:U
Gsequential_batch_normalization_fusedbatchnormv3_readvariableop_resource:W
Isequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource:L
2sequential_conv2d_1_conv2d_readvariableop_resource:	A
3sequential_conv2d_1_biasadd_readvariableop_resource:	

identity_1??>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp?@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?-sequential/batch_normalization/ReadVariableOp?/sequential/batch_normalization/ReadVariableOp_1?(sequential/conv2d/BiasAdd/ReadVariableOp?'sequential/conv2d/Conv2D/ReadVariableOp?*sequential/conv2d_1/BiasAdd/ReadVariableOp?)sequential/conv2d_1/Conv2D/ReadVariableOp_
IdentityIdentityx*
T0*1
_output_shapes
:???????????2

Identity?
'sequential/conv2d/Conv2D/ReadVariableOpReadVariableOp0sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02)
'sequential/conv2d/Conv2D/ReadVariableOp?
sequential/conv2d/Conv2DConv2DIdentity:output:0/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
sequential/conv2d/Conv2D?
(sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOp1sequential_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(sequential/conv2d/BiasAdd/ReadVariableOp?
sequential/conv2d/BiasAddBiasAdd!sequential/conv2d/Conv2D:output:00sequential/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
sequential/conv2d/BiasAdd?
-sequential/batch_normalization/ReadVariableOpReadVariableOp6sequential_batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential/batch_normalization/ReadVariableOp?
/sequential/batch_normalization/ReadVariableOp_1ReadVariableOp8sequential_batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype021
/sequential/batch_normalization/ReadVariableOp_1?
>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpGsequential_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02@
>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp?
@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIsequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02B
@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?
/sequential/batch_normalization/FusedBatchNormV3FusedBatchNormV3"sequential/conv2d/BiasAdd:output:05sequential/batch_normalization/ReadVariableOp:value:07sequential/batch_normalization/ReadVariableOp_1:value:0Fsequential/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Hsequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( 21
/sequential/batch_normalization/FusedBatchNormV3?
sequential/re_lu/ReluRelu3sequential/batch_normalization/FusedBatchNormV3:y:0*
T0*1
_output_shapes
:???????????2
sequential/re_lu/Relu?
)sequential/conv2d_1/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:	*
dtype02+
)sequential/conv2d_1/Conv2D/ReadVariableOp?
sequential/conv2d_1/Conv2DConv2D#sequential/re_lu/Relu:activations:01sequential/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????	*
paddingVALID*
strides
2
sequential/conv2d_1/Conv2D?
*sequential/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02,
*sequential/conv2d_1/BiasAdd/ReadVariableOp?
sequential/conv2d_1/BiasAddBiasAdd#sequential/conv2d_1/Conv2D:output:02sequential/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????	2
sequential/conv2d_1/BiasAddr
reshape/ShapeShape$sequential/conv2d_1/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape/Shape?
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack?
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1?
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2?
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_sliceu
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :?2
reshape/Reshape/shape/1u
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :?2
reshape/Reshape/shape/2t
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :	2
reshape/Reshape/shape/3t
reshape/Reshape/shape/4Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/4t
reshape/Reshape/shape/5Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/5?
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0 reshape/Reshape/shape/4:output:0 reshape/Reshape/shape/5:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape?
reshape/ReshapeReshape$sequential/conv2d_1/BiasAdd:output:0reshape/Reshape/shape:output:0*
T0*9
_output_shapes'
%:#???????????	2
reshape/Reshape?
ExtractImagePatchesExtractImagePatchesx*
T0*1
_output_shapes
:???????????*
ksizes
*
paddingSAME*
rates
*
strides
2
ExtractImagePatcheso
reshape_1/ShapeShapeExtractImagePatches:patches:0*
T0*
_output_shapes
:2
reshape_1/Shape?
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_1/strided_slice/stack?
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_1?
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_2?
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_1/strided_slicey
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :?2
reshape_1/Reshape/shape/1y
reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :?2
reshape_1/Reshape/shape/2x
reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :	2
reshape_1/Reshape/shape/3x
reshape_1/Reshape/shape/4Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/4x
reshape_1/Reshape/shape/5Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/5?
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0"reshape_1/Reshape/shape/2:output:0"reshape_1/Reshape/shape/3:output:0"reshape_1/Reshape/shape/4:output:0"reshape_1/Reshape/shape/5:output:0*
N*
T0*
_output_shapes
:2
reshape_1/Reshape/shape?
reshape_1/ReshapeReshapeExtractImagePatches:patches:0 reshape_1/Reshape/shape:output:0*
T0*9
_output_shapes'
%:#???????????	2
reshape_1/Reshape?
MulMulreshape/Reshape:output:0reshape_1/Reshape:output:0*
T0*9
_output_shapes'
%:#???????????	2
Mulp
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indicesz
SumSumMul:z:0Sum/reduction_indices:output:0*
T0*5
_output_shapes#
!:???????????2
Sum^
reshape_2/ShapeShapeSum:output:0*
T0*
_output_shapes
:2
reshape_2/Shape?
reshape_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_2/strided_slice/stack?
reshape_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_2/strided_slice/stack_1?
reshape_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_2/strided_slice/stack_2?
reshape_2/strided_sliceStridedSlicereshape_2/Shape:output:0&reshape_2/strided_slice/stack:output:0(reshape_2/strided_slice/stack_1:output:0(reshape_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_2/strided_slicey
reshape_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :?2
reshape_2/Reshape/shape/1y
reshape_2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :?2
reshape_2/Reshape/shape/2x
reshape_2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_2/Reshape/shape/3?
reshape_2/Reshape/shapePack reshape_2/strided_slice:output:0"reshape_2/Reshape/shape/1:output:0"reshape_2/Reshape/shape/2:output:0"reshape_2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_2/Reshape/shape?
reshape_2/ReshapeReshapeSum:output:0 reshape_2/Reshape/shape:output:0*
T0*1
_output_shapes
:???????????2
reshape_2/Reshape?

Identity_1Identityreshape_2/Reshape:output:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity_1?
NoOpNoOp?^sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpA^sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1.^sequential/batch_normalization/ReadVariableOp0^sequential/batch_normalization/ReadVariableOp_1)^sequential/conv2d/BiasAdd/ReadVariableOp(^sequential/conv2d/Conv2D/ReadVariableOp+^sequential/conv2d_1/BiasAdd/ReadVariableOp*^sequential/conv2d_1/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:???????????: : : : : : : : 2?
>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp2?
@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_12^
-sequential/batch_normalization/ReadVariableOp-sequential/batch_normalization/ReadVariableOp2b
/sequential/batch_normalization/ReadVariableOp_1/sequential/batch_normalization/ReadVariableOp_12T
(sequential/conv2d/BiasAdd/ReadVariableOp(sequential/conv2d/BiasAdd/ReadVariableOp2R
'sequential/conv2d/Conv2D/ReadVariableOp'sequential/conv2d/Conv2D/ReadVariableOp2X
*sequential/conv2d_1/BiasAdd/ReadVariableOp*sequential/conv2d_1/BiasAdd/ReadVariableOp2V
)sequential/conv2d_1/Conv2D/ReadVariableOp)sequential/conv2d_1/Conv2D/ReadVariableOp:T P
1
_output_shapes
:???????????

_user_specified_namex
?/
?
F__inference_sequential_layer_call_and_return_conditional_losses_307162

inputs?
%conv2d_conv2d_readvariableop_resource:4
&conv2d_biasadd_readvariableop_resource:9
+batch_normalization_readvariableop_resource:;
-batch_normalization_readvariableop_1_resource:J
<batch_normalization_fusedbatchnormv3_readvariableop_resource:L
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource:A
'conv2d_1_conv2d_readvariableop_resource:6
(conv2d_1_biasadd_readvariableop_resource:
identity??"batch_normalization/AssignNewValue?$batch_normalization/AssignNewValue_1?3batch_normalization/FusedBatchNormV3/ReadVariableOp?5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?"batch_normalization/ReadVariableOp?$batch_normalization/ReadVariableOp_1?conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????88*
paddingVALID*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????882
conv2d/BiasAdd?
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype02$
"batch_normalization/ReadVariableOp?
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype02&
$batch_normalization/ReadVariableOp_1?
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOp?
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d/BiasAdd:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????88:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2&
$batch_normalization/FusedBatchNormV3?
"batch_normalization/AssignNewValueAssignVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource1batch_normalization/FusedBatchNormV3:batch_mean:04^batch_normalization/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02$
"batch_normalization/AssignNewValue?
$batch_normalization/AssignNewValue_1AssignVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource5batch_normalization/FusedBatchNormV3:batch_variance:06^batch_normalization/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02&
$batch_normalization/AssignNewValue_1?

re_lu/ReluRelu(batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????882

re_lu/Relu?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2Dre_lu/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????88*
paddingVALID*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????882
conv2d_1/BiasAdd|
IdentityIdentityconv2d_1/BiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????882

Identity?
NoOpNoOp#^batch_normalization/AssignNewValue%^batch_normalization/AssignNewValue_14^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_1^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????88: : : : : : : : 2H
"batch_normalization/AssignNewValue"batch_normalization/AssignNewValue2L
$batch_normalization/AssignNewValue_1$batch_normalization/AssignNewValue_12j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????88
 
_user_specified_nameinputs
?
_
C__inference_re_lu_1_layer_call_and_return_conditional_losses_303920

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????882
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????882

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????88:W S
/
_output_shapes
:?????????88
 
_user_specified_nameinputs
?
?
B__inference_conv2d_layer_call_and_return_conditional_losses_307318

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2	
BiasAddu
IdentityIdentityBiasAdd:output:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?

?
+__inference_sequential_layer_call_fn_307204

inputs!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????88*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_3030972
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????882

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????88: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????88
 
_user_specified_nameinputs
?

?
+__inference_sequential_layer_call_fn_307183

inputs!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????88**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_3029612
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????882

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????88: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????88
 
_user_specified_nameinputs
?
?
B__inference_conv2d_layer_call_and_return_conditional_losses_302458

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????88*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????882	
BiasAdds
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????882

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????88: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????88
 
_user_specified_nameinputs
?

?
+__inference_sequential_layer_call_fn_307069

inputs!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????88**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_sequential_layer_call_and_return_conditional_losses_3025152
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????882

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????88: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????88
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_layer_call_and_return_conditional_losses_307707

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
exponential_avg_factor%
?#<2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
]
A__inference_re_lu_layer_call_and_return_conditional_losses_307628

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????882
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????882

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????88:W S
/
_output_shapes
:?????????88
 
_user_specified_nameinputs
?
?
F__inference_sequential_layer_call_and_return_conditional_losses_303607
conv2d_input'
conv2d_303586:
conv2d_303588:(
batch_normalization_303591:(
batch_normalization_303593:(
batch_normalization_303595:(
batch_normalization_303597:)
conv2d_1_303601:	
conv2d_1_303603:	
identity??+batch_normalization/StatefulPartitionedCall?conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallconv2d_inputconv2d_303586conv2d_303588*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_3033502 
conv2d/StatefulPartitionedCall?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_303591batch_normalization_303593batch_normalization_303595batch_normalization_303597*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_3033732-
+batch_normalization/StatefulPartitionedCall?
re_lu/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_re_lu_layer_call_and_return_conditional_losses_3033882
re_lu/PartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallre_lu/PartitionedCall:output:0conv2d_1_303601conv2d_1_303603*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_3034002"
 conv2d_1/StatefulPartitionedCall?
IdentityIdentity)conv2d_1/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????	2

Identity?
NoOpNoOp,^batch_normalization/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????: : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall:] Y
/
_output_shapes
:?????????
&
_user_specified_nameconv2d_input
?
q
G__inference_concatenate_layer_call_and_return_conditional_losses_303936

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*/
_output_shapes
:?????????882
concatk
IdentityIdentityconcat:output:0*
T0*/
_output_shapes
:?????????882

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:?????????88:?????????88:W S
/
_output_shapes
:?????????88
 
_user_specified_nameinputs:WS
/
_output_shapes
:?????????88
 
_user_specified_nameinputs
?C
?
I__inference_InRFNet_Model_layer_call_and_return_conditional_losses_305177
input_1&
inv_1_305090:
inv_1_305092:
inv_1_305094:
inv_1_305096:
inv_1_305098:
inv_1_305100:&
inv_1_305102:	
inv_1_305104:	'
inv_22_305109:
inv_22_305111:
inv_22_305113:
inv_22_305115:
inv_22_305117:
inv_22_305119:'
inv_22_305121:
inv_22_305123:'
inv_21_305126:
inv_21_305128:
inv_21_305130:
inv_21_305132:
inv_21_305134:
inv_21_305136:'
inv_21_305138:
inv_21_305140:&
inv_3_305147:
inv_3_305149:
inv_3_305151:
inv_3_305153:
inv_3_305155:
inv_3_305157:&
inv_3_305159:	
inv_3_305161:	
dense_305166:	?$@
dense_305168:@ 
dense_1_305171:@
dense_1_305173:
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?inv_1/StatefulPartitionedCall?inv_21/StatefulPartitionedCall?inv_22/StatefulPartitionedCall?inv_3/StatefulPartitionedCall?
inv_1/StatefulPartitionedCallStatefulPartitionedCallinput_1inv_1_305090inv_1_305092inv_1_305094inv_1_305096inv_1_305098inv_1_305100inv_1_305102inv_1_305104*
Tin
2	*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_inv_1_layer_call_and_return_conditional_losses_3046572
inv_1/StatefulPartitionedCall?
re_lu/PartitionedCallPartitionedCall&inv_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_re_lu_layer_call_and_return_conditional_losses_3037312
re_lu/PartitionedCall?
max_pooling2d/PartitionedCallPartitionedCallre_lu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????pp* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_3037372
max_pooling2d/PartitionedCall?
inv_22/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0inv_22_305109inv_22_305111inv_22_305113inv_22_305115inv_22_305117inv_22_305119inv_22_305121inv_22_305123*
Tin
2	*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????88*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_inv_22_layer_call_and_return_conditional_losses_3045332 
inv_22/StatefulPartitionedCall?
inv_21/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0inv_21_305126inv_21_305128inv_21_305130inv_21_305132inv_21_305134inv_21_305136inv_21_305138inv_21_305140*
Tin
2	*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????88*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_inv_21_layer_call_and_return_conditional_losses_3044202 
inv_21/StatefulPartitionedCall?
re_lu_1/PartitionedCallPartitionedCall'inv_21/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????88* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_re_lu_1_layer_call_and_return_conditional_losses_3039202
re_lu_1/PartitionedCall?
re_lu_2/PartitionedCallPartitionedCall'inv_22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????88* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_re_lu_2_layer_call_and_return_conditional_losses_3039272
re_lu_2/PartitionedCall?
concatenate/PartitionedCallPartitionedCall re_lu_1/PartitionedCall:output:0 re_lu_2/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????88* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_concatenate_layer_call_and_return_conditional_losses_3039362
concatenate/PartitionedCall?
max_pooling2d_1/PartitionedCallPartitionedCall$concatenate/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_3039422!
max_pooling2d_1/PartitionedCall?
inv_3/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0inv_3_305147inv_3_305149inv_3_305151inv_3_305153inv_3_305155inv_3_305157inv_3_305159inv_3_305161*
Tin
2	*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_inv_3_layer_call_and_return_conditional_losses_3042832
inv_3/StatefulPartitionedCall?
re_lu_3/PartitionedCallPartitionedCall&inv_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_re_lu_3_layer_call_and_return_conditional_losses_3040372
re_lu_3/PartitionedCall?
flatten/PartitionedCallPartitionedCall re_lu_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????$* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_3040452
flatten/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_305166dense_305168*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_3040582
dense/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_305171dense_1_305173*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_3040752!
dense_1/StatefulPartitionedCall?
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall^inv_1/StatefulPartitionedCall^inv_21/StatefulPartitionedCall^inv_22/StatefulPartitionedCall^inv_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*x
_input_shapesg
e:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2>
inv_1/StatefulPartitionedCallinv_1/StatefulPartitionedCall2@
inv_21/StatefulPartitionedCallinv_21/StatefulPartitionedCall2@
inv_22/StatefulPartitionedCallinv_22/StatefulPartitionedCall2>
inv_3/StatefulPartitionedCallinv_3/StatefulPartitionedCall:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_1
?
?
O__inference_batch_normalization_layer_call_and_return_conditional_losses_301869

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????
 
_user_specified_nameinputs
?
?
.__inference_InRFNet_Model_layer_call_fn_305923

inputs!
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:#
	unknown_5:	
	unknown_6:	#
	unknown_7:
	unknown_8:
	unknown_9:

unknown_10:

unknown_11:

unknown_12:$

unknown_13:

unknown_14:$

unknown_15:

unknown_16:

unknown_17:

unknown_18:

unknown_19:

unknown_20:$

unknown_21:

unknown_22:$

unknown_23:

unknown_24:

unknown_25:

unknown_26:

unknown_27:

unknown_28:$

unknown_29:	

unknown_30:	

unknown_31:	?$@

unknown_32:@

unknown_33:@

unknown_34:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*F
_read_only_resource_inputs(
&$	
 !"#$*0
config_proto 

CPU

GPU2*0J 8? *R
fMRK
I__inference_InRFNet_Model_layer_call_and_return_conditional_losses_3040822
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*x
_input_shapesg
e:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
g
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_303194

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
D__inference_conv2d_1_layer_call_and_return_conditional_losses_303400

inputs8
conv2d_readvariableop_resource:	-
biasadd_readvariableop_resource:	
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????	*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????	2	
BiasAdds
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????	2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
i
M__inference_average_pooling2d_layer_call_and_return_conditional_losses_302302

inputs
identity?
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingSAME*
strides
2	
AvgPool?
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
F__inference_sequential_layer_call_and_return_conditional_losses_303631
conv2d_input'
conv2d_303610:
conv2d_303612:(
batch_normalization_303615:(
batch_normalization_303617:(
batch_normalization_303619:(
batch_normalization_303621:)
conv2d_1_303625:	
conv2d_1_303627:	
identity??+batch_normalization/StatefulPartitionedCall?conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallconv2d_inputconv2d_303610conv2d_303612*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_3033502 
conv2d/StatefulPartitionedCall?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0batch_normalization_303615batch_normalization_303617batch_normalization_303619batch_normalization_303621*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *X
fSRQ
O__inference_batch_normalization_layer_call_and_return_conditional_losses_3034752-
+batch_normalization/StatefulPartitionedCall?
re_lu/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_re_lu_layer_call_and_return_conditional_losses_3033882
re_lu/PartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCallre_lu/PartitionedCall:output:0conv2d_1_303625conv2d_1_303627*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_3034002"
 conv2d_1/StatefulPartitionedCall?
IdentityIdentity)conv2d_1/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????	2

Identity?
NoOpNoOp,^batch_normalization/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????: : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall:] Y
/
_output_shapes
:?????????
&
_user_specified_nameconv2d_input
?
?
D__inference_conv2d_1_layer_call_and_return_conditional_losses_307643

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????88*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????882	
BiasAdds
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????882

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????88: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????88
 
_user_specified_nameinputs
?
?
O__inference_batch_normalization_layer_call_and_return_conditional_losses_307381

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1t
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*M
_output_shapes;
9:???????????:::::*
epsilon%o?:*
is_training( 2
FusedBatchNormV3y
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:???????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?`
?
B__inference_inv_22_layer_call_and_return_conditional_losses_306464
xJ
0sequential_conv2d_conv2d_readvariableop_resource:?
1sequential_conv2d_biasadd_readvariableop_resource:D
6sequential_batch_normalization_readvariableop_resource:F
8sequential_batch_normalization_readvariableop_1_resource:U
Gsequential_batch_normalization_fusedbatchnormv3_readvariableop_resource:W
Isequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource:L
2sequential_conv2d_1_conv2d_readvariableop_resource:A
3sequential_conv2d_1_biasadd_readvariableop_resource:
identity??>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp?@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?-sequential/batch_normalization/ReadVariableOp?/sequential/batch_normalization/ReadVariableOp_1?(sequential/conv2d/BiasAdd/ReadVariableOp?'sequential/conv2d/Conv2D/ReadVariableOp?*sequential/conv2d_1/BiasAdd/ReadVariableOp?)sequential/conv2d_1/Conv2D/ReadVariableOp?
average_pooling2d/AvgPoolAvgPoolx*
T0*/
_output_shapes
:?????????88*
ksize
*
paddingSAME*
strides
2
average_pooling2d/AvgPool?
'sequential/conv2d/Conv2D/ReadVariableOpReadVariableOp0sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02)
'sequential/conv2d/Conv2D/ReadVariableOp?
sequential/conv2d/Conv2DConv2D"average_pooling2d/AvgPool:output:0/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????88*
paddingVALID*
strides
2
sequential/conv2d/Conv2D?
(sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOp1sequential_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(sequential/conv2d/BiasAdd/ReadVariableOp?
sequential/conv2d/BiasAddBiasAdd!sequential/conv2d/Conv2D:output:00sequential/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????882
sequential/conv2d/BiasAdd?
-sequential/batch_normalization/ReadVariableOpReadVariableOp6sequential_batch_normalization_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential/batch_normalization/ReadVariableOp?
/sequential/batch_normalization/ReadVariableOp_1ReadVariableOp8sequential_batch_normalization_readvariableop_1_resource*
_output_shapes
:*
dtype021
/sequential/batch_normalization/ReadVariableOp_1?
>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpGsequential_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype02@
>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp?
@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIsequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype02B
@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?
/sequential/batch_normalization/FusedBatchNormV3FusedBatchNormV3"sequential/conv2d/BiasAdd:output:05sequential/batch_normalization/ReadVariableOp:value:07sequential/batch_normalization/ReadVariableOp_1:value:0Fsequential/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Hsequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????88:::::*
epsilon%o?:*
is_training( 21
/sequential/batch_normalization/FusedBatchNormV3?
sequential/re_lu/ReluRelu3sequential/batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????882
sequential/re_lu/Relu?
)sequential/conv2d_1/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02+
)sequential/conv2d_1/Conv2D/ReadVariableOp?
sequential/conv2d_1/Conv2DConv2D#sequential/re_lu/Relu:activations:01sequential/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????88*
paddingVALID*
strides
2
sequential/conv2d_1/Conv2D?
*sequential/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02,
*sequential/conv2d_1/BiasAdd/ReadVariableOp?
sequential/conv2d_1/BiasAddBiasAdd#sequential/conv2d_1/Conv2D:output:02sequential/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????882
sequential/conv2d_1/BiasAddr
reshape/ShapeShape$sequential/conv2d_1/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape/Shape?
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack?
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1?
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2?
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slicet
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :82
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :82
reshape/Reshape/shape/2t
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/3t
reshape/Reshape/shape/4Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/4t
reshape/Reshape/shape/5Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/5?
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0 reshape/Reshape/shape/4:output:0 reshape/Reshape/shape/5:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape?
reshape/ReshapeReshape$sequential/conv2d_1/BiasAdd:output:0reshape/Reshape/shape:output:0*
T0*7
_output_shapes%
#:!?????????882
reshape/Reshape?
ExtractImagePatchesExtractImagePatchesx*
T0*/
_output_shapes
:?????????88	*
ksizes
*
paddingSAME*
rates
*
strides
2
ExtractImagePatcheso
reshape_1/ShapeShapeExtractImagePatches:patches:0*
T0*
_output_shapes
:2
reshape_1/Shape?
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_1/strided_slice/stack?
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_1?
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_2?
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_1/strided_slicex
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :82
reshape_1/Reshape/shape/1x
reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :82
reshape_1/Reshape/shape/2x
reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/3x
reshape_1/Reshape/shape/4Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/4x
reshape_1/Reshape/shape/5Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/5?
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0"reshape_1/Reshape/shape/2:output:0"reshape_1/Reshape/shape/3:output:0"reshape_1/Reshape/shape/4:output:0"reshape_1/Reshape/shape/5:output:0*
N*
T0*
_output_shapes
:2
reshape_1/Reshape/shape?
reshape_1/ReshapeReshapeExtractImagePatches:patches:0 reshape_1/Reshape/shape:output:0*
T0*7
_output_shapes%
#:!?????????882
reshape_1/Reshape?
MulMulreshape/Reshape:output:0reshape_1/Reshape:output:0*
T0*7
_output_shapes%
#:!?????????882
Mulp
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indicesx
SumSumMul:z:0Sum/reduction_indices:output:0*
T0*3
_output_shapes!
:?????????882
Sum^
reshape_2/ShapeShapeSum:output:0*
T0*
_output_shapes
:2
reshape_2/Shape?
reshape_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_2/strided_slice/stack?
reshape_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_2/strided_slice/stack_1?
reshape_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_2/strided_slice/stack_2?
reshape_2/strided_sliceStridedSlicereshape_2/Shape:output:0&reshape_2/strided_slice/stack:output:0(reshape_2/strided_slice/stack_1:output:0(reshape_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_2/strided_slicex
reshape_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :82
reshape_2/Reshape/shape/1x
reshape_2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :82
reshape_2/Reshape/shape/2x
reshape_2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_2/Reshape/shape/3?
reshape_2/Reshape/shapePack reshape_2/strided_slice:output:0"reshape_2/Reshape/shape/1:output:0"reshape_2/Reshape/shape/2:output:0"reshape_2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_2/Reshape/shape?
reshape_2/ReshapeReshapeSum:output:0 reshape_2/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????882
reshape_2/Reshape}
IdentityIdentityreshape_2/Reshape:output:0^NoOp*
T0*/
_output_shapes
:?????????882

Identity?
NoOpNoOp?^sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpA^sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1.^sequential/batch_normalization/ReadVariableOp0^sequential/batch_normalization/ReadVariableOp_1)^sequential/conv2d/BiasAdd/ReadVariableOp(^sequential/conv2d/Conv2D/ReadVariableOp+^sequential/conv2d_1/BiasAdd/ReadVariableOp*^sequential/conv2d_1/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????pp: : : : : : : : 2?
>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp2?
@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_12^
-sequential/batch_normalization/ReadVariableOp-sequential/batch_normalization/ReadVariableOp2b
/sequential/batch_normalization/ReadVariableOp_1/sequential/batch_normalization/ReadVariableOp_12T
(sequential/conv2d/BiasAdd/ReadVariableOp(sequential/conv2d/BiasAdd/ReadVariableOp2R
'sequential/conv2d/Conv2D/ReadVariableOp'sequential/conv2d/Conv2D/ReadVariableOp2X
*sequential/conv2d_1/BiasAdd/ReadVariableOp*sequential/conv2d_1/BiasAdd/ReadVariableOp2V
)sequential/conv2d_1/Conv2D/ReadVariableOp)sequential/conv2d_1/Conv2D/ReadVariableOp:R N
/
_output_shapes
:?????????pp

_user_specified_namex"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
E
input_1:
serving_default_input_1:0???????????;
dense_10
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
layer_with_weights-1
layer-4
layer_with_weights-2
layer-5
layer-6
layer-7
	layer-8

layer-9
layer_with_weights-3
layer-10
layer-11
layer-12
layer_with_weights-4
layer-13
layer_with_weights-5
layer-14
	optimizer
trainable_variables
regularization_losses
	variables
	keras_api

signatures
+?&call_and_return_all_conditional_losses
?_default_save_signature
?__call__"
_tf_keras_network
"
_tf_keras_input_layer
?

kernel_gen
kernel_reshape
input_patches_reshape
output_reshape
trainable_variables
regularization_losses
	variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
trainable_variables
regularization_losses
 	variables
!	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
"trainable_variables
#regularization_losses
$	variables
%	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
&stride_layer
'
kernel_gen
(kernel_reshape
)input_patches_reshape
*output_reshape
+trainable_variables
,regularization_losses
-	variables
.	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
/stride_layer
0
kernel_gen
1kernel_reshape
2input_patches_reshape
3output_reshape
4trainable_variables
5regularization_losses
6	variables
7	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
8trainable_variables
9regularization_losses
:	variables
;	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
<trainable_variables
=regularization_losses
>	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
@trainable_variables
Aregularization_losses
B	variables
C	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
Dtrainable_variables
Eregularization_losses
F	variables
G	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
H
kernel_gen
Ikernel_reshape
Jinput_patches_reshape
Koutput_reshape
Ltrainable_variables
Mregularization_losses
N	variables
O	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
Ptrainable_variables
Qregularization_losses
R	variables
S	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
Ttrainable_variables
Uregularization_losses
V	variables
W	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

Xkernel
Ybias
Ztrainable_variables
[regularization_losses
\	variables
]	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

^kernel
_bias
`trainable_variables
aregularization_losses
b	variables
c	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
diter

ebeta_1

fbeta_2
	gdecay
hlearning_rateXm?Ym?^m?_m?im?jm?km?lm?mm?nm?om?pm?qm?rm?sm?tm?um?vm?wm?xm?ym?zm?{m?|m?}m?~m?m?	?m?Xv?Yv?^v?_v?iv?jv?kv?lv?mv?nv?ov?pv?qv?rv?sv?tv?uv?vv?wv?xv?yv?zv?{v?|v?}v?~v?v?	?v?"
	optimizer
?
i0
j1
k2
l3
m4
n5
o6
p7
q8
r9
s10
t11
u12
v13
w14
x15
y16
z17
{18
|19
}20
~21
22
?23
X24
Y25
^26
_27"
trackable_list_wrapper
 "
trackable_list_wrapper
?
i0
j1
k2
l3
m4
n5
?6
?7
o8
p9
q10
r11
s12
t13
?14
?15
u16
v17
w18
x19
y20
z21
?22
?23
{24
|25
}26
~27
28
?29
?30
?31
X32
Y33
^34
_35"
trackable_list_wrapper
?
trainable_variables
?layer_metrics
regularization_losses
?metrics
 ?layer_regularization_losses
?layers
	variables
?non_trainable_variables
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
?
?layer_with_weights-0
?layer-0
?layer_with_weights-1
?layer-1
?layer-2
?layer_with_weights-2
?layer-3
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_sequential
?
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
J
i0
j1
k2
l3
m4
n5"
trackable_list_wrapper
 "
trackable_list_wrapper
Z
i0
j1
k2
l3
m4
n5
?6
?7"
trackable_list_wrapper
?
trainable_variables
?layer_metrics
regularization_losses
?metrics
 ?layer_regularization_losses
?layers
	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
trainable_variables
?layer_metrics
regularization_losses
?metrics
 ?layer_regularization_losses
?layers
 	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
"trainable_variables
?layer_metrics
#regularization_losses
?metrics
 ?layer_regularization_losses
?layers
$	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?layer_with_weights-0
?layer-0
?layer_with_weights-1
?layer-1
?layer-2
?layer_with_weights-2
?layer-3
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_sequential
?
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
J
o0
p1
q2
r3
s4
t5"
trackable_list_wrapper
 "
trackable_list_wrapper
Z
o0
p1
q2
r3
s4
t5
?6
?7"
trackable_list_wrapper
?
+trainable_variables
?layer_metrics
,regularization_losses
?metrics
 ?layer_regularization_losses
?layers
-	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?layer_with_weights-0
?layer-0
?layer_with_weights-1
?layer-1
?layer-2
?layer_with_weights-2
?layer-3
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_sequential
?
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
J
u0
v1
w2
x3
y4
z5"
trackable_list_wrapper
 "
trackable_list_wrapper
Z
u0
v1
w2
x3
y4
z5
?6
?7"
trackable_list_wrapper
?
4trainable_variables
?layer_metrics
5regularization_losses
?metrics
 ?layer_regularization_losses
?layers
6	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
8trainable_variables
?layer_metrics
9regularization_losses
?metrics
 ?layer_regularization_losses
?layers
:	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
<trainable_variables
?layer_metrics
=regularization_losses
?metrics
 ?layer_regularization_losses
?layers
>	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
@trainable_variables
?layer_metrics
Aregularization_losses
?metrics
 ?layer_regularization_losses
?layers
B	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Dtrainable_variables
?layer_metrics
Eregularization_losses
?metrics
 ?layer_regularization_losses
?layers
F	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?layer_with_weights-0
?layer-0
?layer_with_weights-1
?layer-1
?layer-2
?layer_with_weights-2
?layer-3
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_sequential
?
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
K
{0
|1
}2
~3
4
?5"
trackable_list_wrapper
 "
trackable_list_wrapper
[
{0
|1
}2
~3
4
?5
?6
?7"
trackable_list_wrapper
?
Ltrainable_variables
?layer_metrics
Mregularization_losses
?metrics
 ?layer_regularization_losses
?layers
N	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Ptrainable_variables
?layer_metrics
Qregularization_losses
?metrics
 ?layer_regularization_losses
?layers
R	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Ttrainable_variables
?layer_metrics
Uregularization_losses
?metrics
 ?layer_regularization_losses
?layers
V	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	?$@2dense/kernel
:@2
dense/bias
.
X0
Y1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
X0
Y1"
trackable_list_wrapper
?
Ztrainable_variables
?layer_metrics
[regularization_losses
?metrics
 ?layer_regularization_losses
?layers
\	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 :@2dense_1/kernel
:2dense_1/bias
.
^0
_1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
^0
_1"
trackable_list_wrapper
?
`trainable_variables
?layer_metrics
aregularization_losses
?metrics
 ?layer_regularization_losses
?layers
b	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
':%2conv2d/kernel
:2conv2d/bias
':%2batch_normalization/gamma
&:$2batch_normalization/beta
):'	2conv2d_1/kernel
:	2conv2d_1/bias
':%2conv2d/kernel
:2conv2d/bias
':%2batch_normalization/gamma
&:$2batch_normalization/beta
):'2conv2d_1/kernel
:2conv2d_1/bias
':%2conv2d/kernel
:2conv2d/bias
':%2batch_normalization/gamma
&:$2batch_normalization/beta
):'2conv2d_1/kernel
:2conv2d_1/bias
':%2conv2d/kernel
:2conv2d/bias
':%2batch_normalization/gamma
&:$2batch_normalization/beta
):'	2conv2d_1/kernel
:	2conv2d_1/bias
/:- (2batch_normalization/moving_mean
3:1 (2#batch_normalization/moving_variance
/:- (2batch_normalization/moving_mean
3:1 (2#batch_normalization/moving_variance
/:- (2batch_normalization/moving_mean
3:1 (2#batch_normalization/moving_variance
/:- (2batch_normalization/moving_mean
3:1 (2#batch_normalization/moving_variance
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14"
trackable_list_wrapper
`
?0
?1
?2
?3
?4
?5
?6
?7"
trackable_list_wrapper
?

ikernel
jbias
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
	?axis
	kgamma
lbeta
?moving_mean
?moving_variance
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

mkernel
nbias
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
J
i0
j1
k2
l3
m4
n5"
trackable_list_wrapper
 "
trackable_list_wrapper
Z
i0
j1
k2
l3
?4
?5
m6
n7"
trackable_list_wrapper
?
?trainable_variables
?layer_metrics
?regularization_losses
?metrics
 ?layer_regularization_losses
?layers
?	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
?layer_metrics
?regularization_losses
?metrics
 ?layer_regularization_losses
?layers
?	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
?layer_metrics
?regularization_losses
?metrics
 ?layer_regularization_losses
?layers
?	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
?layer_metrics
?regularization_losses
?metrics
 ?layer_regularization_losses
?layers
?	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
?layer_metrics
?regularization_losses
?metrics
 ?layer_regularization_losses
?layers
?	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

okernel
pbias
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
	?axis
	qgamma
rbeta
?moving_mean
?moving_variance
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

skernel
tbias
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
J
o0
p1
q2
r3
s4
t5"
trackable_list_wrapper
 "
trackable_list_wrapper
Z
o0
p1
q2
r3
?4
?5
s6
t7"
trackable_list_wrapper
?
?trainable_variables
?layer_metrics
?regularization_losses
?metrics
 ?layer_regularization_losses
?layers
?	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
?layer_metrics
?regularization_losses
?metrics
 ?layer_regularization_losses
?layers
?	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
?layer_metrics
?regularization_losses
?metrics
 ?layer_regularization_losses
?layers
?	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
?layer_metrics
?regularization_losses
?metrics
 ?layer_regularization_losses
?layers
?	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
C
&0
'1
(2
)3
*4"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
?layer_metrics
?regularization_losses
?metrics
 ?layer_regularization_losses
?layers
?	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

ukernel
vbias
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
	?axis
	wgamma
xbeta
?moving_mean
?moving_variance
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

ykernel
zbias
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
J
u0
v1
w2
x3
y4
z5"
trackable_list_wrapper
 "
trackable_list_wrapper
Z
u0
v1
w2
x3
?4
?5
y6
z7"
trackable_list_wrapper
?
?trainable_variables
?layer_metrics
?regularization_losses
?metrics
 ?layer_regularization_losses
?layers
?	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
?layer_metrics
?regularization_losses
?metrics
 ?layer_regularization_losses
?layers
?	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
?layer_metrics
?regularization_losses
?metrics
 ?layer_regularization_losses
?layers
?	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
?layer_metrics
?regularization_losses
?metrics
 ?layer_regularization_losses
?layers
?	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
C
/0
01
12
23
34"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

{kernel
|bias
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
	?axis
	}gamma
~beta
?moving_mean
?moving_variance
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
?

kernel
	?bias
?trainable_variables
?regularization_losses
?	variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"
_tf_keras_layer
K
{0
|1
}2
~3
4
?5"
trackable_list_wrapper
 "
trackable_list_wrapper
[
{0
|1
}2
~3
?4
?5
6
?7"
trackable_list_wrapper
?
?trainable_variables
?layer_metrics
?regularization_losses
?metrics
 ?layer_regularization_losses
?layers
?	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
?layer_metrics
?regularization_losses
?metrics
 ?layer_regularization_losses
?layers
?	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
?layer_metrics
?regularization_losses
?metrics
 ?layer_regularization_losses
?layers
?	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
?layer_metrics
?regularization_losses
?metrics
 ?layer_regularization_losses
?layers
?	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
<
H0
I1
J2
K3"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
R

?total

?count
?	variables
?	keras_api"
_tf_keras_metric
c

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"
_tf_keras_metric
.
i0
j1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
i0
j1"
trackable_list_wrapper
?
?trainable_variables
?layer_metrics
?regularization_losses
?metrics
 ?layer_regularization_losses
?layers
?	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
k0
l1"
trackable_list_wrapper
 "
trackable_list_wrapper
>
k0
l1
?2
?3"
trackable_list_wrapper
?
?trainable_variables
?layer_metrics
?regularization_losses
?metrics
 ?layer_regularization_losses
?layers
?	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
?layer_metrics
?regularization_losses
?metrics
 ?layer_regularization_losses
?layers
?	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
m0
n1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
m0
n1"
trackable_list_wrapper
?
?trainable_variables
?layer_metrics
?regularization_losses
?metrics
 ?layer_regularization_losses
?layers
?	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
o0
p1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
o0
p1"
trackable_list_wrapper
?
?trainable_variables
?layer_metrics
?regularization_losses
?metrics
 ?layer_regularization_losses
?layers
?	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
q0
r1"
trackable_list_wrapper
 "
trackable_list_wrapper
>
q0
r1
?2
?3"
trackable_list_wrapper
?
?trainable_variables
?layer_metrics
?regularization_losses
?metrics
 ?layer_regularization_losses
?layers
?	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
?layer_metrics
?regularization_losses
?metrics
 ?layer_regularization_losses
?layers
?	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
s0
t1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
s0
t1"
trackable_list_wrapper
?
?trainable_variables
?layer_metrics
?regularization_losses
?metrics
 ?layer_regularization_losses
?layers
?	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
u0
v1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
u0
v1"
trackable_list_wrapper
?
?trainable_variables
?layer_metrics
?regularization_losses
?metrics
 ?layer_regularization_losses
?layers
?	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
w0
x1"
trackable_list_wrapper
 "
trackable_list_wrapper
>
w0
x1
?2
?3"
trackable_list_wrapper
?
?trainable_variables
?layer_metrics
?regularization_losses
?metrics
 ?layer_regularization_losses
?layers
?	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
?layer_metrics
?regularization_losses
?metrics
 ?layer_regularization_losses
?layers
?	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
y0
z1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
y0
z1"
trackable_list_wrapper
?
?trainable_variables
?layer_metrics
?regularization_losses
?metrics
 ?layer_regularization_losses
?layers
?	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
{0
|1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
{0
|1"
trackable_list_wrapper
?
?trainable_variables
?layer_metrics
?regularization_losses
?metrics
 ?layer_regularization_losses
?layers
?	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
}0
~1"
trackable_list_wrapper
 "
trackable_list_wrapper
>
}0
~1
?2
?3"
trackable_list_wrapper
?
?trainable_variables
?layer_metrics
?regularization_losses
?metrics
 ?layer_regularization_losses
?layers
?	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
?layer_metrics
?regularization_losses
?metrics
 ?layer_regularization_losses
?layers
?	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
/
0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
/
0
?1"
trackable_list_wrapper
?
?trainable_variables
?layer_metrics
?regularization_losses
?metrics
 ?layer_regularization_losses
?layers
?	variables
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
$:"	?$@2Adam/dense/kernel/m
:@2Adam/dense/bias/m
%:#@2Adam/dense_1/kernel/m
:2Adam/dense_1/bias/m
,:*2Adam/conv2d/kernel/m
:2Adam/conv2d/bias/m
,:*2 Adam/batch_normalization/gamma/m
+:)2Adam/batch_normalization/beta/m
.:,	2Adam/conv2d_1/kernel/m
 :	2Adam/conv2d_1/bias/m
,:*2Adam/conv2d/kernel/m
:2Adam/conv2d/bias/m
,:*2 Adam/batch_normalization/gamma/m
+:)2Adam/batch_normalization/beta/m
.:,2Adam/conv2d_1/kernel/m
 :2Adam/conv2d_1/bias/m
,:*2Adam/conv2d/kernel/m
:2Adam/conv2d/bias/m
,:*2 Adam/batch_normalization/gamma/m
+:)2Adam/batch_normalization/beta/m
.:,2Adam/conv2d_1/kernel/m
 :2Adam/conv2d_1/bias/m
,:*2Adam/conv2d/kernel/m
:2Adam/conv2d/bias/m
,:*2 Adam/batch_normalization/gamma/m
+:)2Adam/batch_normalization/beta/m
.:,	2Adam/conv2d_1/kernel/m
 :	2Adam/conv2d_1/bias/m
$:"	?$@2Adam/dense/kernel/v
:@2Adam/dense/bias/v
%:#@2Adam/dense_1/kernel/v
:2Adam/dense_1/bias/v
,:*2Adam/conv2d/kernel/v
:2Adam/conv2d/bias/v
,:*2 Adam/batch_normalization/gamma/v
+:)2Adam/batch_normalization/beta/v
.:,	2Adam/conv2d_1/kernel/v
 :	2Adam/conv2d_1/bias/v
,:*2Adam/conv2d/kernel/v
:2Adam/conv2d/bias/v
,:*2 Adam/batch_normalization/gamma/v
+:)2Adam/batch_normalization/beta/v
.:,2Adam/conv2d_1/kernel/v
 :2Adam/conv2d_1/bias/v
,:*2Adam/conv2d/kernel/v
:2Adam/conv2d/bias/v
,:*2 Adam/batch_normalization/gamma/v
+:)2Adam/batch_normalization/beta/v
.:,2Adam/conv2d_1/kernel/v
 :2Adam/conv2d_1/bias/v
,:*2Adam/conv2d/kernel/v
:2Adam/conv2d/bias/v
,:*2 Adam/batch_normalization/gamma/v
+:)2Adam/batch_normalization/beta/v
.:,	2Adam/conv2d_1/kernel/v
 :	2Adam/conv2d_1/bias/v
?2?
I__inference_InRFNet_Model_layer_call_and_return_conditional_losses_305554
I__inference_InRFNet_Model_layer_call_and_return_conditional_losses_305846
I__inference_InRFNet_Model_layer_call_and_return_conditional_losses_305087
I__inference_InRFNet_Model_layer_call_and_return_conditional_losses_305177?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
!__inference__wrapped_model_301847input_1"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_InRFNet_Model_layer_call_fn_304157
.__inference_InRFNet_Model_layer_call_fn_305923
.__inference_InRFNet_Model_layer_call_fn_306000
.__inference_InRFNet_Model_layer_call_fn_304997?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
A__inference_inv_1_layer_call_and_return_conditional_losses_306070
A__inference_inv_1_layer_call_and_return_conditional_losses_306140?
???
FullArgSpec$
args?
jself
jx

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_inv_1_layer_call_fn_306161
&__inference_inv_1_layer_call_fn_306182?
???
FullArgSpec$
args?
jself
jx

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_re_lu_layer_call_and_return_conditional_losses_306187?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_re_lu_layer_call_fn_306192?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_306197
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_306202?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_max_pooling2d_layer_call_fn_306207
.__inference_max_pooling2d_layer_call_fn_306212?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_inv_21_layer_call_and_return_conditional_losses_306282
B__inference_inv_21_layer_call_and_return_conditional_losses_306352?
???
FullArgSpec$
args?
jself
jx

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_inv_21_layer_call_fn_306373
'__inference_inv_21_layer_call_fn_306394?
???
FullArgSpec$
args?
jself
jx

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_inv_22_layer_call_and_return_conditional_losses_306464
B__inference_inv_22_layer_call_and_return_conditional_losses_306534?
???
FullArgSpec$
args?
jself
jx

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_inv_22_layer_call_fn_306555
'__inference_inv_22_layer_call_fn_306576?
???
FullArgSpec$
args?
jself
jx

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_re_lu_1_layer_call_and_return_conditional_losses_306581?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_re_lu_1_layer_call_fn_306586?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_re_lu_2_layer_call_and_return_conditional_losses_306591?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_re_lu_2_layer_call_fn_306596?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_concatenate_layer_call_and_return_conditional_losses_306603?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_concatenate_layer_call_fn_306609?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_306614
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_306619?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
0__inference_max_pooling2d_1_layer_call_fn_306624
0__inference_max_pooling2d_1_layer_call_fn_306629?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_inv_3_layer_call_and_return_conditional_losses_306699
A__inference_inv_3_layer_call_and_return_conditional_losses_306769?
???
FullArgSpec$
args?
jself
jx

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_inv_3_layer_call_fn_306790
&__inference_inv_3_layer_call_fn_306811?
???
FullArgSpec$
args?
jself
jx

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_re_lu_3_layer_call_and_return_conditional_losses_306816?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_re_lu_3_layer_call_fn_306821?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_flatten_layer_call_and_return_conditional_losses_306827?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_flatten_layer_call_fn_306832?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_dense_layer_call_and_return_conditional_losses_306843?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_dense_layer_call_fn_306852?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_1_layer_call_and_return_conditional_losses_306863?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_1_layer_call_fn_306872?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
$__inference_signature_wrapper_305262input_1"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_sequential_layer_call_and_return_conditional_losses_306903
F__inference_sequential_layer_call_and_return_conditional_losses_306934
F__inference_sequential_layer_call_and_return_conditional_losses_302247
F__inference_sequential_layer_call_and_return_conditional_losses_302271?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
+__inference_sequential_layer_call_fn_302066
+__inference_sequential_layer_call_fn_306955
+__inference_sequential_layer_call_fn_306976
+__inference_sequential_layer_call_fn_302223?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
M__inference_average_pooling2d_layer_call_and_return_conditional_losses_306981?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
2__inference_average_pooling2d_layer_call_fn_306986?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_sequential_layer_call_and_return_conditional_losses_307017
F__inference_sequential_layer_call_and_return_conditional_losses_307048
F__inference_sequential_layer_call_and_return_conditional_losses_302715
F__inference_sequential_layer_call_and_return_conditional_losses_302739?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
+__inference_sequential_layer_call_fn_302534
+__inference_sequential_layer_call_fn_307069
+__inference_sequential_layer_call_fn_307090
+__inference_sequential_layer_call_fn_302691?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
M__inference_average_pooling2d_layer_call_and_return_conditional_losses_307095?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
2__inference_average_pooling2d_layer_call_fn_307100?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_sequential_layer_call_and_return_conditional_losses_307131
F__inference_sequential_layer_call_and_return_conditional_losses_307162
F__inference_sequential_layer_call_and_return_conditional_losses_303161
F__inference_sequential_layer_call_and_return_conditional_losses_303185?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
+__inference_sequential_layer_call_fn_302980
+__inference_sequential_layer_call_fn_307183
+__inference_sequential_layer_call_fn_307204
+__inference_sequential_layer_call_fn_303137?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_sequential_layer_call_and_return_conditional_losses_307235
F__inference_sequential_layer_call_and_return_conditional_losses_307266
F__inference_sequential_layer_call_and_return_conditional_losses_303607
F__inference_sequential_layer_call_and_return_conditional_losses_303631?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
+__inference_sequential_layer_call_fn_303426
+__inference_sequential_layer_call_fn_307287
+__inference_sequential_layer_call_fn_307308
+__inference_sequential_layer_call_fn_303583?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_conv2d_layer_call_and_return_conditional_losses_307318?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_conv2d_layer_call_fn_307327?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
O__inference_batch_normalization_layer_call_and_return_conditional_losses_307345
O__inference_batch_normalization_layer_call_and_return_conditional_losses_307363
O__inference_batch_normalization_layer_call_and_return_conditional_losses_307381
O__inference_batch_normalization_layer_call_and_return_conditional_losses_307399?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
4__inference_batch_normalization_layer_call_fn_307412
4__inference_batch_normalization_layer_call_fn_307425
4__inference_batch_normalization_layer_call_fn_307438
4__inference_batch_normalization_layer_call_fn_307451?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
A__inference_re_lu_layer_call_and_return_conditional_losses_307456?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_re_lu_layer_call_fn_307461?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_conv2d_1_layer_call_and_return_conditional_losses_307471?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_conv2d_1_layer_call_fn_307480?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_conv2d_layer_call_and_return_conditional_losses_307490?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_conv2d_layer_call_fn_307499?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
O__inference_batch_normalization_layer_call_and_return_conditional_losses_307517
O__inference_batch_normalization_layer_call_and_return_conditional_losses_307535
O__inference_batch_normalization_layer_call_and_return_conditional_losses_307553
O__inference_batch_normalization_layer_call_and_return_conditional_losses_307571?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
4__inference_batch_normalization_layer_call_fn_307584
4__inference_batch_normalization_layer_call_fn_307597
4__inference_batch_normalization_layer_call_fn_307610
4__inference_batch_normalization_layer_call_fn_307623?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
A__inference_re_lu_layer_call_and_return_conditional_losses_307628?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_re_lu_layer_call_fn_307633?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_conv2d_1_layer_call_and_return_conditional_losses_307643?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_conv2d_1_layer_call_fn_307652?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_conv2d_layer_call_and_return_conditional_losses_307662?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_conv2d_layer_call_fn_307671?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
O__inference_batch_normalization_layer_call_and_return_conditional_losses_307689
O__inference_batch_normalization_layer_call_and_return_conditional_losses_307707
O__inference_batch_normalization_layer_call_and_return_conditional_losses_307725
O__inference_batch_normalization_layer_call_and_return_conditional_losses_307743?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
4__inference_batch_normalization_layer_call_fn_307756
4__inference_batch_normalization_layer_call_fn_307769
4__inference_batch_normalization_layer_call_fn_307782
4__inference_batch_normalization_layer_call_fn_307795?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
A__inference_re_lu_layer_call_and_return_conditional_losses_307800?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_re_lu_layer_call_fn_307805?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_conv2d_1_layer_call_and_return_conditional_losses_307815?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_conv2d_1_layer_call_fn_307824?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_conv2d_layer_call_and_return_conditional_losses_307834?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_conv2d_layer_call_fn_307843?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
O__inference_batch_normalization_layer_call_and_return_conditional_losses_307861
O__inference_batch_normalization_layer_call_and_return_conditional_losses_307879
O__inference_batch_normalization_layer_call_and_return_conditional_losses_307897
O__inference_batch_normalization_layer_call_and_return_conditional_losses_307915?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
4__inference_batch_normalization_layer_call_fn_307928
4__inference_batch_normalization_layer_call_fn_307941
4__inference_batch_normalization_layer_call_fn_307954
4__inference_batch_normalization_layer_call_fn_307967?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
A__inference_re_lu_layer_call_and_return_conditional_losses_307972?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_re_lu_layer_call_fn_307977?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_conv2d_1_layer_call_and_return_conditional_losses_307987?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_conv2d_1_layer_call_fn_307996?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
I__inference_InRFNet_Model_layer_call_and_return_conditional_losses_305087?-ijkl??mnuvwx??yzopqr??st{|}~???XY^_B??
8?5
+?(
input_1???????????
p 

 
? "%?"
?
0?????????
? ?
I__inference_InRFNet_Model_layer_call_and_return_conditional_losses_305177?-ijkl??mnuvwx??yzopqr??st{|}~???XY^_B??
8?5
+?(
input_1???????????
p

 
? "%?"
?
0?????????
? ?
I__inference_InRFNet_Model_layer_call_and_return_conditional_losses_305554?-ijkl??mnuvwx??yzopqr??st{|}~???XY^_A?>
7?4
*?'
inputs???????????
p 

 
? "%?"
?
0?????????
? ?
I__inference_InRFNet_Model_layer_call_and_return_conditional_losses_305846?-ijkl??mnuvwx??yzopqr??st{|}~???XY^_A?>
7?4
*?'
inputs???????????
p

 
? "%?"
?
0?????????
? ?
.__inference_InRFNet_Model_layer_call_fn_304157?-ijkl??mnuvwx??yzopqr??st{|}~???XY^_B??
8?5
+?(
input_1???????????
p 

 
? "???????????
.__inference_InRFNet_Model_layer_call_fn_304997?-ijkl??mnuvwx??yzopqr??st{|}~???XY^_B??
8?5
+?(
input_1???????????
p

 
? "???????????
.__inference_InRFNet_Model_layer_call_fn_305923?-ijkl??mnuvwx??yzopqr??st{|}~???XY^_A?>
7?4
*?'
inputs???????????
p 

 
? "???????????
.__inference_InRFNet_Model_layer_call_fn_306000?-ijkl??mnuvwx??yzopqr??st{|}~???XY^_A?>
7?4
*?'
inputs???????????
p

 
? "???????????
!__inference__wrapped_model_301847?-ijkl??mnuvwx??yzopqr??st{|}~???XY^_:?7
0?-
+?(
input_1???????????
? "1?.
,
dense_1!?
dense_1??????????
M__inference_average_pooling2d_layer_call_and_return_conditional_losses_306981?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
M__inference_average_pooling2d_layer_call_and_return_conditional_losses_307095?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
2__inference_average_pooling2d_layer_call_fn_306986?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
2__inference_average_pooling2d_layer_call_fn_307100?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
O__inference_batch_normalization_layer_call_and_return_conditional_losses_307345?kl??M?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
O__inference_batch_normalization_layer_call_and_return_conditional_losses_307363?kl??M?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
O__inference_batch_normalization_layer_call_and_return_conditional_losses_307381xkl??=?:
3?0
*?'
inputs???????????
p 
? "/?,
%?"
0???????????
? ?
O__inference_batch_normalization_layer_call_and_return_conditional_losses_307399xkl??=?:
3?0
*?'
inputs???????????
p
? "/?,
%?"
0???????????
? ?
O__inference_batch_normalization_layer_call_and_return_conditional_losses_307517?qr??M?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
O__inference_batch_normalization_layer_call_and_return_conditional_losses_307535?qr??M?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
O__inference_batch_normalization_layer_call_and_return_conditional_losses_307553tqr??;?8
1?.
(?%
inputs?????????88
p 
? "-?*
#? 
0?????????88
? ?
O__inference_batch_normalization_layer_call_and_return_conditional_losses_307571tqr??;?8
1?.
(?%
inputs?????????88
p
? "-?*
#? 
0?????????88
? ?
O__inference_batch_normalization_layer_call_and_return_conditional_losses_307689?wx??M?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
O__inference_batch_normalization_layer_call_and_return_conditional_losses_307707?wx??M?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
O__inference_batch_normalization_layer_call_and_return_conditional_losses_307725twx??;?8
1?.
(?%
inputs?????????88
p 
? "-?*
#? 
0?????????88
? ?
O__inference_batch_normalization_layer_call_and_return_conditional_losses_307743twx??;?8
1?.
(?%
inputs?????????88
p
? "-?*
#? 
0?????????88
? ?
O__inference_batch_normalization_layer_call_and_return_conditional_losses_307861?}~??M?J
C?@
:?7
inputs+???????????????????????????
p 
? "??<
5?2
0+???????????????????????????
? ?
O__inference_batch_normalization_layer_call_and_return_conditional_losses_307879?}~??M?J
C?@
:?7
inputs+???????????????????????????
p
? "??<
5?2
0+???????????????????????????
? ?
O__inference_batch_normalization_layer_call_and_return_conditional_losses_307897t}~??;?8
1?.
(?%
inputs?????????
p 
? "-?*
#? 
0?????????
? ?
O__inference_batch_normalization_layer_call_and_return_conditional_losses_307915t}~??;?8
1?.
(?%
inputs?????????
p
? "-?*
#? 
0?????????
? ?
4__inference_batch_normalization_layer_call_fn_307412?kl??M?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
4__inference_batch_normalization_layer_call_fn_307425?kl??M?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
4__inference_batch_normalization_layer_call_fn_307438kkl??=?:
3?0
*?'
inputs???????????
p 
? ""?????????????
4__inference_batch_normalization_layer_call_fn_307451kkl??=?:
3?0
*?'
inputs???????????
p
? ""?????????????
4__inference_batch_normalization_layer_call_fn_307584?qr??M?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
4__inference_batch_normalization_layer_call_fn_307597?qr??M?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
4__inference_batch_normalization_layer_call_fn_307610gqr??;?8
1?.
(?%
inputs?????????88
p 
? " ??????????88?
4__inference_batch_normalization_layer_call_fn_307623gqr??;?8
1?.
(?%
inputs?????????88
p
? " ??????????88?
4__inference_batch_normalization_layer_call_fn_307756?wx??M?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
4__inference_batch_normalization_layer_call_fn_307769?wx??M?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
4__inference_batch_normalization_layer_call_fn_307782gwx??;?8
1?.
(?%
inputs?????????88
p 
? " ??????????88?
4__inference_batch_normalization_layer_call_fn_307795gwx??;?8
1?.
(?%
inputs?????????88
p
? " ??????????88?
4__inference_batch_normalization_layer_call_fn_307928?}~??M?J
C?@
:?7
inputs+???????????????????????????
p 
? "2?/+????????????????????????????
4__inference_batch_normalization_layer_call_fn_307941?}~??M?J
C?@
:?7
inputs+???????????????????????????
p
? "2?/+????????????????????????????
4__inference_batch_normalization_layer_call_fn_307954g}~??;?8
1?.
(?%
inputs?????????
p 
? " ???????????
4__inference_batch_normalization_layer_call_fn_307967g}~??;?8
1?.
(?%
inputs?????????
p
? " ???????????
G__inference_concatenate_layer_call_and_return_conditional_losses_306603?j?g
`?]
[?X
*?'
inputs/0?????????88
*?'
inputs/1?????????88
? "-?*
#? 
0?????????88
? ?
,__inference_concatenate_layer_call_fn_306609?j?g
`?]
[?X
*?'
inputs/0?????????88
*?'
inputs/1?????????88
? " ??????????88?
D__inference_conv2d_1_layer_call_and_return_conditional_losses_307471pmn9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????	
? ?
D__inference_conv2d_1_layer_call_and_return_conditional_losses_307643lst7?4
-?*
(?%
inputs?????????88
? "-?*
#? 
0?????????88
? ?
D__inference_conv2d_1_layer_call_and_return_conditional_losses_307815lyz7?4
-?*
(?%
inputs?????????88
? "-?*
#? 
0?????????88
? ?
D__inference_conv2d_1_layer_call_and_return_conditional_losses_307987m?7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????	
? ?
)__inference_conv2d_1_layer_call_fn_307480cmn9?6
/?,
*?'
inputs???????????
? ""????????????	?
)__inference_conv2d_1_layer_call_fn_307652_st7?4
-?*
(?%
inputs?????????88
? " ??????????88?
)__inference_conv2d_1_layer_call_fn_307824_yz7?4
-?*
(?%
inputs?????????88
? " ??????????88?
)__inference_conv2d_1_layer_call_fn_307996`?7?4
-?*
(?%
inputs?????????
? " ??????????	?
B__inference_conv2d_layer_call_and_return_conditional_losses_307318pij9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
B__inference_conv2d_layer_call_and_return_conditional_losses_307490lop7?4
-?*
(?%
inputs?????????88
? "-?*
#? 
0?????????88
? ?
B__inference_conv2d_layer_call_and_return_conditional_losses_307662luv7?4
-?*
(?%
inputs?????????88
? "-?*
#? 
0?????????88
? ?
B__inference_conv2d_layer_call_and_return_conditional_losses_307834l{|7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
'__inference_conv2d_layer_call_fn_307327cij9?6
/?,
*?'
inputs???????????
? ""?????????????
'__inference_conv2d_layer_call_fn_307499_op7?4
-?*
(?%
inputs?????????88
? " ??????????88?
'__inference_conv2d_layer_call_fn_307671_uv7?4
-?*
(?%
inputs?????????88
? " ??????????88?
'__inference_conv2d_layer_call_fn_307843_{|7?4
-?*
(?%
inputs?????????
? " ???????????
C__inference_dense_1_layer_call_and_return_conditional_losses_306863\^_/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????
? {
(__inference_dense_1_layer_call_fn_306872O^_/?,
%?"
 ?
inputs?????????@
? "???????????
A__inference_dense_layer_call_and_return_conditional_losses_306843]XY0?-
&?#
!?
inputs??????????$
? "%?"
?
0?????????@
? z
&__inference_dense_layer_call_fn_306852PXY0?-
&?#
!?
inputs??????????$
? "??????????@?
C__inference_flatten_layer_call_and_return_conditional_losses_306827a7?4
-?*
(?%
inputs?????????
? "&?#
?
0??????????$
? ?
(__inference_flatten_layer_call_fn_306832T7?4
-?*
(?%
inputs?????????
? "???????????$?
A__inference_inv_1_layer_call_and_return_conditional_losses_306070w
ijkl??mn8?5
.?+
%?"
x???????????
p 
? "/?,
%?"
0???????????
? ?
A__inference_inv_1_layer_call_and_return_conditional_losses_306140w
ijkl??mn8?5
.?+
%?"
x???????????
p
? "/?,
%?"
0???????????
? ?
&__inference_inv_1_layer_call_fn_306161j
ijkl??mn8?5
.?+
%?"
x???????????
p 
? ""?????????????
&__inference_inv_1_layer_call_fn_306182j
ijkl??mn8?5
.?+
%?"
x???????????
p
? ""?????????????
B__inference_inv_21_layer_call_and_return_conditional_losses_306282s
opqr??st6?3
,?)
#? 
x?????????pp
p 
? "-?*
#? 
0?????????88
? ?
B__inference_inv_21_layer_call_and_return_conditional_losses_306352s
opqr??st6?3
,?)
#? 
x?????????pp
p
? "-?*
#? 
0?????????88
? ?
'__inference_inv_21_layer_call_fn_306373f
opqr??st6?3
,?)
#? 
x?????????pp
p 
? " ??????????88?
'__inference_inv_21_layer_call_fn_306394f
opqr??st6?3
,?)
#? 
x?????????pp
p
? " ??????????88?
B__inference_inv_22_layer_call_and_return_conditional_losses_306464s
uvwx??yz6?3
,?)
#? 
x?????????pp
p 
? "-?*
#? 
0?????????88
? ?
B__inference_inv_22_layer_call_and_return_conditional_losses_306534s
uvwx??yz6?3
,?)
#? 
x?????????pp
p
? "-?*
#? 
0?????????88
? ?
'__inference_inv_22_layer_call_fn_306555f
uvwx??yz6?3
,?)
#? 
x?????????pp
p 
? " ??????????88?
'__inference_inv_22_layer_call_fn_306576f
uvwx??yz6?3
,?)
#? 
x?????????pp
p
? " ??????????88?
A__inference_inv_3_layer_call_and_return_conditional_losses_306699t{|}~???6?3
,?)
#? 
x?????????
p 
? "-?*
#? 
0?????????
? ?
A__inference_inv_3_layer_call_and_return_conditional_losses_306769t{|}~???6?3
,?)
#? 
x?????????
p
? "-?*
#? 
0?????????
? ?
&__inference_inv_3_layer_call_fn_306790g{|}~???6?3
,?)
#? 
x?????????
p 
? " ???????????
&__inference_inv_3_layer_call_fn_306811g{|}~???6?3
,?)
#? 
x?????????
p
? " ???????????
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_306614?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
K__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_306619h7?4
-?*
(?%
inputs?????????88
? "-?*
#? 
0?????????
? ?
0__inference_max_pooling2d_1_layer_call_fn_306624?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
0__inference_max_pooling2d_1_layer_call_fn_306629[7?4
-?*
(?%
inputs?????????88
? " ???????????
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_306197?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_306202j9?6
/?,
*?'
inputs???????????
? "-?*
#? 
0?????????pp
? ?
.__inference_max_pooling2d_layer_call_fn_306207?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
.__inference_max_pooling2d_layer_call_fn_306212]9?6
/?,
*?'
inputs???????????
? " ??????????pp?
C__inference_re_lu_1_layer_call_and_return_conditional_losses_306581h7?4
-?*
(?%
inputs?????????88
? "-?*
#? 
0?????????88
? ?
(__inference_re_lu_1_layer_call_fn_306586[7?4
-?*
(?%
inputs?????????88
? " ??????????88?
C__inference_re_lu_2_layer_call_and_return_conditional_losses_306591h7?4
-?*
(?%
inputs?????????88
? "-?*
#? 
0?????????88
? ?
(__inference_re_lu_2_layer_call_fn_306596[7?4
-?*
(?%
inputs?????????88
? " ??????????88?
C__inference_re_lu_3_layer_call_and_return_conditional_losses_306816h7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
(__inference_re_lu_3_layer_call_fn_306821[7?4
-?*
(?%
inputs?????????
? " ???????????
A__inference_re_lu_layer_call_and_return_conditional_losses_306187l9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
A__inference_re_lu_layer_call_and_return_conditional_losses_307456l9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
A__inference_re_lu_layer_call_and_return_conditional_losses_307628h7?4
-?*
(?%
inputs?????????88
? "-?*
#? 
0?????????88
? ?
A__inference_re_lu_layer_call_and_return_conditional_losses_307800h7?4
-?*
(?%
inputs?????????88
? "-?*
#? 
0?????????88
? ?
A__inference_re_lu_layer_call_and_return_conditional_losses_307972h7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????
? ?
&__inference_re_lu_layer_call_fn_306192_9?6
/?,
*?'
inputs???????????
? ""?????????????
&__inference_re_lu_layer_call_fn_307461_9?6
/?,
*?'
inputs???????????
? ""?????????????
&__inference_re_lu_layer_call_fn_307633[7?4
-?*
(?%
inputs?????????88
? " ??????????88?
&__inference_re_lu_layer_call_fn_307805[7?4
-?*
(?%
inputs?????????88
? " ??????????88?
&__inference_re_lu_layer_call_fn_307977[7?4
-?*
(?%
inputs?????????
? " ???????????
F__inference_sequential_layer_call_and_return_conditional_losses_302247?
ijkl??mnG?D
=?:
0?-
conv2d_input???????????
p 

 
? "/?,
%?"
0???????????	
? ?
F__inference_sequential_layer_call_and_return_conditional_losses_302271?
ijkl??mnG?D
=?:
0?-
conv2d_input???????????
p

 
? "/?,
%?"
0???????????	
? ?
F__inference_sequential_layer_call_and_return_conditional_losses_302715?
opqr??stE?B
;?8
.?+
conv2d_input?????????88
p 

 
? "-?*
#? 
0?????????88
? ?
F__inference_sequential_layer_call_and_return_conditional_losses_302739?
opqr??stE?B
;?8
.?+
conv2d_input?????????88
p

 
? "-?*
#? 
0?????????88
? ?
F__inference_sequential_layer_call_and_return_conditional_losses_303161?
uvwx??yzE?B
;?8
.?+
conv2d_input?????????88
p 

 
? "-?*
#? 
0?????????88
? ?
F__inference_sequential_layer_call_and_return_conditional_losses_303185?
uvwx??yzE?B
;?8
.?+
conv2d_input?????????88
p

 
? "-?*
#? 
0?????????88
? ?
F__inference_sequential_layer_call_and_return_conditional_losses_303607?{|}~???E?B
;?8
.?+
conv2d_input?????????
p 

 
? "-?*
#? 
0?????????	
? ?
F__inference_sequential_layer_call_and_return_conditional_losses_303631?{|}~???E?B
;?8
.?+
conv2d_input?????????
p

 
? "-?*
#? 
0?????????	
? ?
F__inference_sequential_layer_call_and_return_conditional_losses_306903?
ijkl??mnA?>
7?4
*?'
inputs???????????
p 

 
? "/?,
%?"
0???????????	
? ?
F__inference_sequential_layer_call_and_return_conditional_losses_306934?
ijkl??mnA?>
7?4
*?'
inputs???????????
p

 
? "/?,
%?"
0???????????	
? ?
F__inference_sequential_layer_call_and_return_conditional_losses_307017|
opqr??st??<
5?2
(?%
inputs?????????88
p 

 
? "-?*
#? 
0?????????88
? ?
F__inference_sequential_layer_call_and_return_conditional_losses_307048|
opqr??st??<
5?2
(?%
inputs?????????88
p

 
? "-?*
#? 
0?????????88
? ?
F__inference_sequential_layer_call_and_return_conditional_losses_307131|
uvwx??yz??<
5?2
(?%
inputs?????????88
p 

 
? "-?*
#? 
0?????????88
? ?
F__inference_sequential_layer_call_and_return_conditional_losses_307162|
uvwx??yz??<
5?2
(?%
inputs?????????88
p

 
? "-?*
#? 
0?????????88
? ?
F__inference_sequential_layer_call_and_return_conditional_losses_307235}{|}~?????<
5?2
(?%
inputs?????????
p 

 
? "-?*
#? 
0?????????	
? ?
F__inference_sequential_layer_call_and_return_conditional_losses_307266}{|}~?????<
5?2
(?%
inputs?????????
p

 
? "-?*
#? 
0?????????	
? ?
+__inference_sequential_layer_call_fn_302066y
ijkl??mnG?D
=?:
0?-
conv2d_input???????????
p 

 
? ""????????????	?
+__inference_sequential_layer_call_fn_302223y
ijkl??mnG?D
=?:
0?-
conv2d_input???????????
p

 
? ""????????????	?
+__inference_sequential_layer_call_fn_302534u
opqr??stE?B
;?8
.?+
conv2d_input?????????88
p 

 
? " ??????????88?
+__inference_sequential_layer_call_fn_302691u
opqr??stE?B
;?8
.?+
conv2d_input?????????88
p

 
? " ??????????88?
+__inference_sequential_layer_call_fn_302980u
uvwx??yzE?B
;?8
.?+
conv2d_input?????????88
p 

 
? " ??????????88?
+__inference_sequential_layer_call_fn_303137u
uvwx??yzE?B
;?8
.?+
conv2d_input?????????88
p

 
? " ??????????88?
+__inference_sequential_layer_call_fn_303426v{|}~???E?B
;?8
.?+
conv2d_input?????????
p 

 
? " ??????????	?
+__inference_sequential_layer_call_fn_303583v{|}~???E?B
;?8
.?+
conv2d_input?????????
p

 
? " ??????????	?
+__inference_sequential_layer_call_fn_306955s
ijkl??mnA?>
7?4
*?'
inputs???????????
p 

 
? ""????????????	?
+__inference_sequential_layer_call_fn_306976s
ijkl??mnA?>
7?4
*?'
inputs???????????
p

 
? ""????????????	?
+__inference_sequential_layer_call_fn_307069o
opqr??st??<
5?2
(?%
inputs?????????88
p 

 
? " ??????????88?
+__inference_sequential_layer_call_fn_307090o
opqr??st??<
5?2
(?%
inputs?????????88
p

 
? " ??????????88?
+__inference_sequential_layer_call_fn_307183o
uvwx??yz??<
5?2
(?%
inputs?????????88
p 

 
? " ??????????88?
+__inference_sequential_layer_call_fn_307204o
uvwx??yz??<
5?2
(?%
inputs?????????88
p

 
? " ??????????88?
+__inference_sequential_layer_call_fn_307287p{|}~?????<
5?2
(?%
inputs?????????
p 

 
? " ??????????	?
+__inference_sequential_layer_call_fn_307308p{|}~?????<
5?2
(?%
inputs?????????
p

 
? " ??????????	?
$__inference_signature_wrapper_305262?-ijkl??mnuvwx??yzopqr??st{|}~???XY^_E?B
? 
;?8
6
input_1+?(
input_1???????????"1?.
,
dense_1!?
dense_1?????????