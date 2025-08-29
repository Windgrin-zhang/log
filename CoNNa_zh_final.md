<div class="page-break">

</div>

Microprocessors and Microsystems 73 （2020） 102991

![](/workspace/CoNNa_zh_media/89259587f36c5dd19cbb8a85f1befaa825a57c23.jpg)

![](/workspace/CoNNa_zh_media/4d6dbf964160164d8dbdb7ba8ba50e3930991f41.jpg)

## Contents lists available at ScienceDirect

## Microprocessors and Microsystems

journal homepage： www. elsevier. com/locate/micpro

CoNNa–Hardware accelerator for compressed convolutional neural

![](/workspace/CoNNa_zh_media/f94f3990da74cfce53ec99231e3bbcdf6e67db88.png)

networks

∗ a ， b a Rastislav J. R. Struharik ， Bogdan Z. Vukobratovi ´c ，
Andrea M. Erdeljan ，

a ´c Damjan M. Rakanovi

a University of Novi Sad， Faculty of Technical Sciences， Trg Dositeja
Obradovi ´ca 6， Novi Sad， 210 0 0， Serbia

b Kortiq GmbH， Gebrüder-Eicher-Ring 45， Forstern， Germany

a r t i c l e i n f o a b s t r a c t

Article history： In this paper， we propose a novel 卷积神经网络
hardware accelerator called CoNNA， ca-

Received 6 January 2019 pable of accelerating pruned， quantized CNNs.
In contrast to most existing solutions， CoNNA offers a

Revised 21 December 2019

complete solution to the compressed 卷积神经网络 acceleration， being
able to accelerate all 层 types commonly

Accepted 2 January 2020

found in contemporary CNNs. CoNNA is designed as a coarse-grained
reconﬁgurable 结构， which

Available online 11 January 2020

uses rapid， dynamic reconﬁguration during 卷积神经网络 层 processing.
The CoNNA 结构 enables the

on-the-ﬂy selection of the 卷积神经网络 network that should be
accelerated and also supports the acceleration of Keywords：

卷积神经网络 networks with dynamic topology. Furthermore， by being able
to directly process compressed 特征 Machine learning

卷积神经网络 and kernel maps， and skip all ineffectual com putations
during 卷积神经网络 层 processing， the CoNNA 卷积神经网络 ac-

卷积神经网络 pruning celerator is able to achieve higher 卷积神经网络
processing rates than some of the previously proposed solutions。

compressed 卷积神经网络 + The CoNNA 结构 has been implemented using
Xilinx ZynqUtrascale FPGA family and compared

## Hardware acceleration

with seven previously proposed 卷积神经网络 hardware accelerators.
Results of the experiments seem to indicate

## FPGA

that the CoNNA 结构 is up to 14. 10， 6. 05， 4. 91， 2. 67， 11. 30，
3. 08 and 3. 58 times faster than previ-

ously proposed MIT’s Eyeriss， NullHop， NVIDIA’s Deep Learning
Accelerator （NVDLA）， NEURAghe， CNN_A1，

fpgaConvNet， and Deephi’s Aristotle 卷积神经网络 accelerators
respectively， while using identical number of com-

puting units and operating at the same clock frequency。

© 2020 Elsevier B. V. All rights reserved。

1\. Introduction to obtain an effective representation of input space.
This approach

is different from earlier attempts that have used manually crafted

Deep learning \[1\] ， and particularly Deep Neural Networks features or
rules designed by experts. Because of this CNNs cur-

（DNNs）， are currently one of the most intensively and widely used
rently offer the best recognition quality versus alternative object

machine learning predictive models. DNNs are not a new concept
recognition or image classiﬁcation algorithms。

\[2\] ， but after recent breakthrough applications in the ﬁelds of im-
However， the superior 准确率 of CNNs comes at a high cost

age processing \[3–5\] . and speech recognition \[6\] ， they have re-
because of their computational and storage complexity. State-of-

turned to the academic and industrial focus. Today， different types
the-art CNNs are described by hundreds of millions of parameters

of DNNs are being employed in a wide range of applications， rang- and
require billions of computations in order to classify single in-

ing from autonomous driving \[7\] ， medical \[8\] ， and even to
playing put instance \[3–5\] . For example， one of the largest
卷积神经网络 networks，

× complex games \[9\] . In many of these application domains， DNNs
VGG-16 卷积神经网络 \[10\] ， which operates on 224 224 input images，
re-

are now able to exceed human levels of performance. The excep- quires
around 500 MB for storing network parameters and per-

tional performance of DNNs， and in particular Convolutional Neu-
forming more than 30 billion ﬂoating-point operations in order to

ral Networks （CNNs） \[3\] ， predominantly arises from their ability
classify single input image. It is highly likely that future CNNs will

to automatically extract high-level features from raw sensory data be
even larger， deeper， will process larger input instances， requir-

during the 训练 phase， using a large amount of data， in order ing even
more computations per input instance， and will be used

to perform more intricate classiﬁcation tasks at faster speeds， ever-

increasingly in real-time， within low-power operating conditions。

∗ Corresponding author. Because of this， careful selection of
appropriate computing plat-

E-mail addresses： rasti@uns. ac. rs （R. J. R. Struharik）， bogdan.
vukobratovic@kortiq. form for the implementation of 卷积神经网络-based
applications is of great

com （B. Z. Vukobratovi ´c）， andrea. erdeljan@uns. ac. rs （A. M.
Erdeljan），

importance。

rdamjan@uns. ac. rs （D. M. Rakanovi ´c）.

https： //doi. org/10. 1016/j. micpro. 2020. 102991

0141-9331/© 2020 Elsevier B. V. All rights reserved。

<div class="page-break">

</div>

## 2 R. J. R. Struharik， B. Z. Vukobratovi ´c and A. M. Erdeljan et al. / Microprocessors and Microsystems 73 （2020） 102991

Currently， there are two approaches to implementing 卷积神经网络 net-
types， types of non-linear 激活 functions， etc. This is one of

works： the main drawbacks of the majority of previously proposed cus-

tom hardware solutions， which puts them at a great disadvantage

## 1 Using multicore processor-based hardware accelerators （CPUs

when compared with CPU/GPU implementations。

or GPUs）

Both groups of accelerators （CPU/GPU and custom） tend to use

## 2 Using dedicated hardware accelerators （ASICs or FPGAs）

a sequential approach when processing the 卷积神经网络 network.
卷积神经网络

network is processed 层 by 层， storing intermediate values The ﬁrst
group of CPU/GPU based accelerators offers highly ﬂex-

in the external， off-chip DRAM memory. Usually， no parallelization
ible and easy to develop solutions， in terms of supported 卷积神经网络

of 层 processing is employed. Please notice， that this frequent
architectures， kernel parameters， nonlinear 激活 functions，

movement of data between the accelerator and DRAM memory is pooling
algorithms， and deep learning software frameworks （Caffe，

one of the most power-consuming operations， and accounts for the
TensorFlow， Keras， Matlab）， but is ineﬃcient when usage of avail-

majority of power consumption of the complete accelerator sys- able
computing resources and power eﬃciency is considered. Stan-

dard CPU can perform between 10–100 GFLOP/s， with the power tem。

Parallelization in both groups of accelerators mostly comes from
consumption that is usually around 100 Watts. Therefore， using

the parallel evaluation of individual convolution operations， located
CPUs in high-performance requirements， found in 卷积神经网络 cloud
appli-

cations， or in low power requirements in 边/mobile applications within
convolutional layers， usually using a 2D array of processing

elements. Multiplications and additions involved in the convolution is
diﬃcult. In contrast， GPUs can reach over 10 TOP/s of peak per-

calculation are executed in parallel， on dedicated multiplier and
formance， but with power consumption reaching over 250 Watts

\[11\] ， GPUs are good choices only for high-performance 卷积神经网络
cloud adder modules located within each processing element. Although

this is an eﬃcient way of achieving higher performance numbers，
applications but are not suitable for the 边/mobile applications。

it has one signiﬁcant drawback. Mapping of convolution operations NVIDIA
is offering its Jetson TX2 family \[12\] based on GPUs with

of different sizes （with different sizes and shapes of convolutional
Pascal 结构 as the 边 solution. TX2 can reach the perfor-

kernels and with different values of kernel stride values） to these
mance of up to 1 TOP/s but with the power consumption of over

2D array structures can be very diﬃcult， and even when it is pos- 10
Watts， which is still too high for most of the 边 applications，

sible typically results in ineﬃcient usage of available multipliers
which can require solutions with the power consumption of less

than 1 Watt. and adders， therefore signiﬁcantly decreasing the eﬃciency
of an

accelerator. Dedicated hardware accelerators offer much higher
utilization

In this paper， we present a novel， coarse-grained reconﬁgurable， of
computing resources， but usually lack ﬂexibility， i. e. they sup-

port only a few different 卷积神经网络 architectures with smaller ranges
compressed 卷积神经网络 hardware accelerator， named CoNNA， which aims

to overcome these diﬃculties. The CoNNA 结构 is based of supported
卷积神经网络 结构 parameters， like kernel size， shape，

on a different way of parallelizing 卷积神经网络 operations. It
sequentially etc. This is mostly the case because solutions from this
group use

highly optimized hardware structures for calculation of only cer-
computes individual convolution operation， keeping it folded， using

a single processing unit to calculate all multiply-accumulate oper- tain
pre-deﬁned convolutional kernel conﬁgurations， which then

ations contained within one convolution， but employs a number cannot be
reconﬁgured to eﬃciently calculate other convolutional

kernel conﬁgurations. However， because of the higher utilization of of
these processing units to compute a number of convolutions in

parallel. This parallelizing approach can easily accommodate con-
available computing resources， accelerators from this group tend to

volutional kernels of different sizes， shapes and stride values， with-
be more power-eﬃcient than the processor-based accelerators。

out signiﬁcant 损失 on the computational eﬃciency. While general-purpose
compute engines， especially GPUs， have

been the mainstay for much of contemporary 卷积神经网络 processing，
Furthermore， CoNNA is designed to be able to directly oper-

ate on compressed 卷积神经网络 networks and compressed 特征 maps，
increasingly there is a growing interest in providing more 卷积神经网络

which results in a signiﬁcant increase in 卷积神经网络 processing
perfor- implementations based on FPGAs. This shift is occurring mainly

because of two reasons. Improvements in FPGA technology re- mance. The
CoNNA 结构 is highly conﬁgurable， enabling

various types of 卷积神经网络 families （VGG， Inception， ResNet，
MobileNet， cently demonstrated FPGA performance which comes very close

NASNet， etc. ） to be implemented eﬃciently. It is also capable of ac-
to GPU performance， with the reported performance of 9. 2 TOP/s

for FPGA \[13\] . Second， recent trends in 卷积神经网络 结构 develop-
celerating complete compressed CNNs， supporting all major layers

found in contemporary CNNs， like convolutional， depthwise convo- ment
increasingly exploit the sparsity of 卷积神经网络 networks and the

lutional， pooling， adding and fully-connected layers. use of extreme
compact data types to represent data that is be-

ing processed. These trends strongly favor FPGA devices， which are The
CoNNA 结构 is not the ﬁrst 结构 that takes

beneﬁt from processing sparse， compressed CNNs. Some of the designed to
easily handle irregular parallelism， which is present

most notable previously proposed compressed 卷积神经网络 accelerators
when working with sparse CNNs， and custom data types. As a re-

are Cnvlutin \[31\] ， NullHop \[32\] ， SparseNN \[33\] ， Cambricon-x
\[34\] ， sult， next-generation 卷积神经网络 accelerators are expected
to deliver up

EIE \[35\] ， and Scnn \[36\] . Although not being the ﬁrst compressed
to x5. 4 better computational throughput than GPUs \[14\] .

卷积神经网络 accelerator， CoNNA still has some important advantages
over The ﬁeld of custom 卷积神经网络 hardware accelerators has been in

all these previously proposed architectures. the focus of the academic
community in recent years， generat-

ing more than ninety different solutions. However， the majority of
Cnvlutin 结构 \[31\] ， proposed by Albericio et al. bene-

ﬁts from the sparsity of input 特征 maps but not the sparsity these
solutions are concerned with acceleration of only one par-

of weights. It is closely based on the well-known dense 卷积神经网络 ac-
ticular 层 type from the 卷积神经网络， typically the convolutional 层，

because of its high computational demand， \[15–19\] . A signiﬁcant
celerator DaDianNao \[15\] ， where the authors of Cnvlutin have re-

designed the DaDianNao’s NFU module and created a CNV module， number of
architectures have been proposed for the acceleration

capable of eﬃcient detection and skipping of zeros present in input of
complete CNNs \[20–30\] . Furthermore， most of the proposed so-

lutions are able to accelerate only uncompressed CNNs \[15–30\] ， 特征
maps. The CoNNA 结构 can also skip input 特征

maps zeros， but it can also skip any zeros that are present in the with
only several examples being able to process compressed CNNs

convolutional kernels， or fully-connected weights， which should re- to
some degree \[31–36\] . In addition， almost all proposed solu-

sult in more eﬃcient 卷积神经网络 processing when compared to Cnvlutin.
tions are not highly conﬁgurable， severely limiting the ﬂexibility of

Furthermore， Cnvlutin is designed to perform complete 卷积神经网络 pro-
supported CNNs， in terms of supported 层 types， kernel recep-

cessing storing all intermediate data in the on-chip memory. This tive
ﬁeld sizes， horizontal and vertical kernel stride values， pooling

<div class="page-break">

</div>

R. J. R. Struharik， B. Z. Vukobratovi ´c and A. M. Erdeljan et al. /
Microprocessors and Microsystems 73 （2020） 102991 3

is a power-eﬃcient solution but requires large on-chip memories， 2.
Compressed convolutional neural networks

of the order of tens of megabytes， and cannot support arbitrarily

large 卷积神经网络 networks. On the other hand， CoNNA stores intermedi-
卷积神经网络 \[37\] is a type of feed-forward ar-

ate 特征 maps in the external DRAM memory， but the power tiﬁcial neural
network in which the connectivity pattern between

consumption is reduced signiﬁcantly because CoNNA uses an on- the
neurons is inspired by the neural connectivity found in the an-

chip 特征 map cache to minimize the amount of data movement imal visual
cortex. Individual neurons from the visual cortex re-

between the accelerator and the external memory. However， by us- spond
to stimuli only from a restricted region of space， known as

ing external memory to store all 卷积神经网络 related data， CoNNA can
pro- the receptive ﬁeld. Receptive ﬁelds of neighboring neurons par-

vide acceleration to arbitrarily sized 卷积神经网络 networks. tially
overlap， spanning the entire visual ﬁeld. Previously it was

NullHop 结构 \[32\] ， proposed by Aimar et al. is similar to shown that
the response of an individual neuron to the stimuli

Cnvlutin since it also takes beneﬁt from skipping any zeros that are
within its receptive ﬁeld can be approximated mathematically by

present in the input 特征 maps but cannot skip zeros in the 卷积神经网络
a convolution operation， which is extensively used in CNNs。

weights. Since the CoNNA 结构 can effectively skip all zeros，
卷积神经网络 结构 is formed by stacking together layers of differ-

present either in the input 特征 maps or in the 卷积神经网络 weights，
entiable functions， which gradually transform input instance into

it should offer a more eﬃcient acceleration of CNNs. Similar to an
appropriate output response （e. g. holding the class scores）.

CoNNA， NullHop stores both 卷积神经网络 weights and compressed inter- A
number of different layers types are commonly used when

mediate 特征 maps in external memory， but NullHop doesn’t building a
卷积神经网络： convolutional 层， depthwise convolutional 层，

implement any input 特征 map caching， which CoNNA does， to pooling
层， non-linear 激活 层， adding 层， concatena-

reduce the required data movement between the accelerator and tion 层，
and fully-connected 层。

external memory when processing convolutional and pooling lay- Before
proceeding， let us introduce some notations. By an input

ers. 特征 map （IFM） we will assume a 3D set of values that make

SparseNN \[33\] and Cambricon-x \[34\] architectures take the op- the
input volume of the current 卷积神经网络 层. The input values vol-

posite approach than one taken in Cnvlutin and NullHop archi- ume of the
ﬁrst 卷积神经网络 层 is usually not called an input 特征

tectures， by skipping zeros that are present in the 卷积神经网络
weights map； rather it is called an input instance. Similarly， an
output fea-

but are not able to skip zeros that are present in the input fea- ture
map （OFM） designates the 3D set of 激活 values of every

ture maps. By being able to skip zeros that are present both neuron
present in the current 卷积神经网络 层. The output volume of the

in input 特征 maps and 卷积神经网络 weights the CoNNA 结构 last
卷积神经网络 层 is usually not called the output 特征 map； rather

should offer more eﬃcient 卷积神经网络 acceleration than Cambricon-x and
it is called the classiﬁcation vector。

× × SparseNN architectures. A 3D region of N M D points within the input
特征 map

Similar to CoNNA， EIE 结构 \[35\] can skip zeros located that is
directly connected to one neuron from the current 卷积神经网络

both in input 特征 maps and 卷积神经网络 weight. But， EIE is able 层
will be designated as the input 特征 map bundle. In most

to do this only for fully-connected layers， and not for convolu-
卷积神经网络 architectures values for the horizontal and vertical size
of the

= tional layers. Therefore， EIE 结构 cannot be used to pro- input 特征
map bundle are equal， N M . For example， the size

cess convolutional， pooling or adding layers from CNNs， which of the
input 特征 map bundle of the 4th convolutional 层

× × makes EIE an interesting solution for the acceleration of fully-
from the VGG-16 卷积神经网络 network \[10\] is a 3D region of 3 3 128

connected 卷积神经网络 layers， but not the complete compressed
卷积神经网络 ac- points. Similarly， the size of the input 特征 map
bundle for the

celerator. On the other hand， the CoNNA 结构 is designed 2nd pooling 层
from the VGG-16 network is a 3D region of

× × to be capable of accelerating all standard 卷积神经网络 层 types，
in- 2 2 1 points。

× × cluding convolutional， depthwise convolutional， pooling， adding A
3D region of 1 1 D points within the 特征 map bundle

and fully-connected layers. Furthermore， during the acceleration will
be designated as the 特征 map stick. If the 特征 map is

of convolutional and fully-connected layers， CoNNA is able to skip the
input 特征 map， then the 特征 map stick will be called

all zeros that are present in the input 特征 maps and convo- the input
特征 map stick. Accordingly， if the 特征 map is the

lutional coeﬃcient and fully-connected weight maps. This makes output
特征 map， then the 特征 map stick will be called the

CoNNA a universal pruned 卷积神经网络 accelerator， as opposed to the
EIE output 特征 map stick. Every input 特征 map bundle is com-

结构. posed of a number of input 特征 map sticks. For example， the

Scnn 结构 \[36\] can also skip all zeros located both in input 特征 map
bundle of the 4th convolutional 层 from the

input 特征 maps and 卷积神经网络 weights， similar to EIE and CoNNA ar-
VGG 卷积神经网络 is composed of 9 input 特征 map sticks， each being a

× × chitectures， but it can process only convolutional 层 type. So the
3D region of 1 1 128 points。

Scnn 结构 is to a degree complementary to the EIE archi- Fig. 1
illustrates the deﬁnitions of the 特征 map stick and the

tecture. Both architectures can skip all zeros but in different 层 特征
map bundle， in the case when the 特征 map bundle is

types. Once more， CoNNA is able to skip all zeros both in convo-
composed of 9 特征 map sticks。

lutional and fully-connected layers， and opposed to EIE and Scnn CNNs
are both computationally and memory demanding， which

结构 offers a solution to hardware acceleration of complete makes their
usage diﬃcult， especially in embedded applications。

pruned CNNs. For example， VGG-16 卷积神经网络 \[10\] has more than 138
million net-

The rest of the paper is organized as follows. Section 2 presents work
parameters. Using 32-bit ﬂoating number representation， ap-

a brief introduction to CNNs， with the emphasis on 卷积神经网络 com-
proximately 552 MB of memory space must be available only for

pression， since CoNNA takes great beneﬁt from 卷积神经网络 compression.
storing all required network parameters. Furthermore， during 层

Section 3 presents details of the proposed CoNNA 结构， de- processing，
卷积神经网络 uses an input 特征 map， which is either the in-

scribing all of its major components. Section 4 contains the re- put
image itself， or the output of the previous 卷积神经网络 层， and pro-

sults of the experiments aimed at comparing the performance of duces an
output 特征 map （OFM）. The sizes of input and output

the CoNNA 结构 with some of the previously proposed 卷积神经网络 特征
maps depend on the characteristics of 卷积神经网络 layers. For ex-

hardware acceleration solutions， and also presents a detailed dis-
ample， the largest input and output 特征 maps in the case of

cussion about the impact of the irregular distribution of non-zero the
VGG-16 卷积神经网络 are more than 12 MB large each， when 32-bit

input 特征 map and 卷积神经网络 weight values on the CoNNA’s
卷积神经网络 ﬂoating-point representation is used. Compared to the size
of the

processing eﬃciency. Section 5 holds ﬁnal remarks and conclu- memory
required to store VGG-16 卷积神经网络 parameters， the size of the

sions. required memory for storing all intermediate 特征 maps is only

<div class="page-break">

</div>

## 4 R. J. R. Struharik， B. Z. Vukobratovi ´c and A. M. Erdeljan et al. / Microprocessors and Microsystems 73 （2020） 102991

Fig. 1. Illustration of deﬁnitions of 特征 map bundle and 特征 map
stick。

around 6% of the required network parameters memory size. How-
rameters， especially if 卷积神经网络 parameters are stored in
external，

ever， the total required 特征 map data movement size to pro- off-chip
memory， which is usually the case。

cess one input image reaches around 120 MB. If we analyze the

A critical step during the 卷积神经网络 network pruning process requires

number of required computations in order to classify one input

identifying kernel coeﬃcients or fully-connected weights that are

image， in the case of VGG-16 卷积神经网络 it reaches over 15 G of MAC

redundant and can be removed. Since this can be done in nu-

operations， or equivalently， over 30 GOPs. For example， if VGG-16

merous ways， a large number of different pruning algorithms have

卷积神经网络 is deployed in a system that is processing real-time video
data

been proposed in the open literature \[38–45\] ， with the new ones

stream， with a speed of 25 frames-per-second， the required com-

continuously being proposed. However， all 卷积神经网络 pruning
algorithms

pute power of 卷积神经网络 implementation system reaches 750 GOP/sec

can be broadly divided into two large categories：

with required data throughput of at least 16. 8 GB/sec. This makes

deployment of CNNs in embedded applications very challenging， 1.
Coarse-grained pruning algorithms – these algorithms remove

especially when latency， throughput and/or power consumption is
complete convolutional kernels （entire ﬁlters） from a convolu-

of interest. tional 层. Some of the previously proposed coarse-grained

One way of solving these issues is to reduce the memory size pruning
algorithms are \[38–41\] .

required to store network parameters and intermediate 特征 2.
Fine-grained pruning algorithms – these algorithms remove in-

maps， by using different 卷积神经网络 compression methods. Generally，
all dividual kernel coeﬃcients from selected ﬁlter within convolu-

卷积神经网络 compression methods can be divided into two groups， de-
tional layers， leaving the original number of ﬁlters unchanged，

pending on the target data： methods that compress 卷积神经网络 network
or remove individual weights from fully-connected layers. Some

parameters， and methods that compress 特征 map data. of the previously
proposed ﬁne-grained pruning algorithms are

卷积神经网络 pruning \[38–45\] is the major technique used to compress
\[42–45\] .

卷积神经网络 network parameters. 卷积神经网络 pruning can be beneﬁcial
because

One widely used 卷积神经网络 network 参数 compression algo-

of a number of reasons：

rithm is the “Deep Compression” algorithm， proposed by Song Han，

1\. By removing redundant weights memory footprint of the target et al.
in \[44\] . “Deep Compression” algorithm uses a three-stage

卷积神经网络 network is reduced， allowing it to be used in memory-
pipeline to reduce the storage size required by the 卷积神经网络
network， in

constrained applications. Since pruned weights are actually set a manner
that preserves original 准确率. First， the original 卷积神经网络

to zero， resulting pruned 卷积神经网络 is a sparse structure，
containing network is pruned by removing all redundant kernel coeﬃcients

many zeros within its convolutional kernel and fully-connected and
fully-connected weights， keeping only the most informative

weight maps. Some of the data compression techniques could ones， using
ﬁne-grained pruning algorithm. Next， remaining con-

be used to compress this sparse representation of 卷积神经网络， which
nection weights are quantized so that multiple connections share

will then require a smaller amount of memory space for its the same
weight， thus only the codebooks （effective weights） and

storage. the indices need to be stored. Finally， Huffman encoding is
applied

2\. Trainable 卷积神经网络 weights are dominantly located within
convolu- to take advantage of the biased distribution of effective
weights。

tional and fully-connected layers and are always used in mul- After
performing all three steps of the “Deep Compression” algo-

tiplicative operations. If some of these weights are set to zero，
rithm， the size of memory required for storing all 卷积神经网络
parameters

because they are redundant， product terms that involve these can be
reduced signiﬁcantly. For example， in the case of VGG-16

zero-valued weights actually don’t have to be computed， be-
卷积神经网络， the required memory size is reduced from 552 MB to only

cause their value will also be equal to zero. If we could detect 5. 5
MB。

these zero-outcome product terms and skip them， the process- Please
notice that pruning and weight quantization affect the

ing time of convolutional or fully-connected layers could be re- overall
准确率 of the 卷积神经网络 network， but as shown in \[44\] ， signif-

duced， resulting in shorter input instance processing time. The icant
pruning and quantization can be applied to standard CNNs，

process of skipping zero-valued product terms is also known as trained
on standard datasets， without degrading initial， unpruned，

the “Zero Skipping” process， and the CoNNA accelerator is de-
卷积神经网络 network 准确率. In this paper， we also use ﬁne-grained

signed to use it during 卷积神经网络 processing. Speeding-up instance
pruning to sparsify 卷积神经网络 network parameters before the sparse

processing by zero skipping can be beneﬁcial because it can ei-
卷积神经网络 network is run on the CoNNA 卷积神经网络 accelerator.
However， we

ther allow us to process input instances at a faster rate， result-
don’t cluster remaining weights into codebooks， nor do we use

ing in higher throughput， or alternatively save power because Huffman
encoding. However， we do perform quantization of un-

we could run 卷积神经网络 accelerator at a slower clock frequency and
pruned weights into a 16-bit ﬁxed-point representation to further

still be able to ﬁnish with instance processing on time. reduce the
memory footprint and enable more eﬃcient FPGA im-

3\. Finally， since the compressed sparse representation of 卷积神经网络
net- plementation。

work is signiﬁcantly smaller in size when compared with the In a
ﬁne-grained pruning approach， each convolutional kernel

original， un-pruned， dense representation of 卷积神经网络， a
signiﬁcant is pruned individually. Usually， a speciﬁed number of kernel
coef-

amount of energy can be saved during movement of 卷积神经网络 pa-
ﬁcients are removed from each kernel， to create a sparse convo-

<div class="page-break">

</div>

R. J. R. Struharik， B. Z. Vukobratovi ´c and A. M. Erdeljan et al. /
Microprocessors and Microsystems 73 （2020） 102991 5

Fig. 2. Fine-grained 卷积神经网络 pruning principle of operation。

lutional kernel. For example， Fig. 2 illustrates the process of ﬁne-
underlying problem that 卷积神经网络 network is being trained to solve，

× × grained pruning of a 3 3 5 convolutional kernel. pruning algorithm
that is being used， distribution of pruning ratios

We start with the dense representation of the convolutional over layers
from 卷积神经网络 network selected for pruning， 优化

kernel， shown on the left of Fig. 2 . Dense kernel coeﬃcients are
algorithm used to train 卷积神经网络 network， etc. Achieving the
highest

obtained during the 训练 process of the selected 卷积神经网络 network.
pruning ratios for the selected layers from the 卷积神经网络 network can

During ﬁne-grained pruning， a speciﬁed number of kernel coef- be a very
time-consuming， trial and error， process。

ﬁcients are removed， effectively setting them to zero. In the ex-
Please notice that pruned， sparse representations of 卷积神经网络 net-

ample from Fig. 2 ， a total of 33 kernel coeﬃcients are removed works
by themselves don’t reduce the memory size required for

× × × × from dense 3 3 5 kernel， to obtain a sparse 3 3 5 ker- storing
network parameters. They also don’t reduce the amount of

nel， shown on the right of Fig. 2 . In this sparse representation of
data that will be moved between the 卷积神经网络 accelerator and operat-

× × 3 3 5 kernel， only 12 coeﬃcients have non-zero values， while ing
memory， nor do they reduce the time required to process the

the remaining 33 coeﬃcients are set to zero. input images. Pruning only
introduces a certain amount of zero-

Criteria upon which kernel coeﬃcients are selected for pruning valued
parameters within the 卷积神经网络 network description， but the

can be numerous. One commonly used approach is to prune ker- total
number of network parameters remains the same as in the

nel coeﬃcients with the smallest absolute values. In this approach，
original unpruned 卷积神经网络 network。

selected kernel coeﬃcients are ﬁrst sorted in ascending order of
Therefore， the amount of memory required to store pruned

their absolute value， and the speciﬁed number of smallest valued
卷积神经网络 network parameters will be identical with the amount re-

coeﬃcients is then set to zero. This process is repeated for all con-
quired to store parameters of the unpruned 卷积神经网络 network， the
only

volutional kernels from all convolutional layers of the target
卷积神经网络 difference being that a certain amount of these parameters
will

network， with possibly different pruning ratios for each individual
have zero value in the case of pruned 卷积神经网络 network. Similarly，
the

convolutional 层. Weight values from fully-connected layers can amount
of data movement between 卷积神经网络 accelerator and the op-

also be pruned using a similar approach in order to create a sparse
erating memory， where the 卷积神经网络 network parameters are stored，

weight map for every fully-connected 层. will be identical for unpruned
and pruned 卷积神经网络 network represen-

After 卷积神经网络 network pruning has been performed the result is a
tations， the difference once more being the fact that in the case of

sparse representation of selected 卷积神经网络 network， containing a
sig- pruned 卷积神经网络 network many of parameters that will be
transferred

niﬁcant number of zero-valued convolutional coeﬃcients or fully- to the
accelerator will be zero-valued。

connected weight values. Usually， especially when high pruning ra-
However， since pruned 卷积神经网络 network representation contains a

tios are employed， the resulting pruned 卷积神经网络 network has
signiﬁ- signiﬁcant number of zero-valued parameters， lossless data com-

cantly lower 准确率， compared with the original dense， unpruned
pression algorithms can be used to reduce the amount of data re-

卷积神经网络 network. Therefore， an additional retraining process of
the quired to store pruned 卷积神经网络 network representation. The
CoNNA

pruned 卷积神经网络 network is necessary， in order to regain the 准确率
结构 uses a variation of standard Zero Run-Length （ZRL）

of the unpruned 卷积神经网络 network. Depending on how severe the
卷积神经网络 encoding compression algorithm \[19\] to compress sparse
pruned

pruning process has been， retraining of pruned 卷积神经网络 network can
卷积神经网络 representations. Using ZRL compression a signiﬁcant reduc-

restore complete or only a fraction of original unpruned 卷积神经网络
net- tion in the memory size required to store pruned 卷积神经网络
network

work 准确率. representation and the total data transfer size required to
transfer

Although 卷积神经网络 network pruning can be performed incremen- pruned
卷积神经网络 network representation to the 卷积神经网络 accelerator，
can be

tally， 层 by 层， most common approach is to prune all applica-
achieved. For more details about the ZRL compression algorithm

ble 卷积神经网络 layers at the same time， create one sparse
representation used within the CoNNA 结构 and the amount of data stor-

of complete 卷积神经网络 network and then retrain it， in order to
regain as age and transfer size reduction that is achievable， please
refer to

much of original 准确率. Please notice that the presented prun- Horowitz
\[46\] .

ing algorithm was used in this paper to prune the 卷积神经网络 networks
Please also notice that， compared to the unpruned 卷积神经网络 net-

prior to their acceleration by the CoNNA 卷积神经网络 accelerator.
work， having a pruned 卷积神经网络 network doesn’t reduce the num-

The amount of pruning that can be used with given 卷积神经网络 net- ber
of MAC operations that need to be performed in order to

work depends on a number of factors： 卷积神经网络 network 结构，
process one input image. The only difference will be that in the

<div class="page-break">

</div>

## 6 R. J. R. Struharik， B. Z. Vukobratovi ´c and A. M. Erdeljan et al. / Microprocessors and Microsystems 73 （2020） 102991

case of a pruned 卷积神经网络 network， a number of these MAC opera-
accelerators proposed in \[31，32\] use this 优化 technique

tions will result in zero value since we will be multiplying in- to
increase 卷积神经网络 processing eﬃciency。

put 特征 map points with zero-valued convolutional coeﬃcients 2. Weights
“zero-skipping” - in this approach all MAC operations

or fully-connected weight values. Reduction in instance processing where
convolutional coeﬃcient or fully-connected weight value

time could be possible if the 卷积神经网络 accelerator could somehow
“de- is zero are skipped during the process of convolutional and

tect” and “skip” these zero-valued MAC operations since they don’t
weighted sum calculation. For example， 卷积神经网络 accelerators pro-

change the ﬁnal outcome of convolution or weighted sum calcula- posed in
\[33，34\] use this 优化 technique to increase

tions. The CoNNA 结构 was designed speciﬁcally to exploit 卷积神经网络
processing eﬃciency。

this 优化 opportunity. 3. “All zero-skipping” - in this approach all MAC
operations where

Reduction in memory size required for storing 卷积神经网络 network
either IFM point or convolutional coeﬃcient/fully-connected

representation， reduction in required data transfer size and reduc-
weight value is zero are skipped during the process of convolu-

tion in instance processing time can have beneﬁcial effects in low-
tional and weighted sum calculation. For example， 卷积神经网络 accel-

ering the cost and increasing the energy eﬃciency of 卷积神经网络
network erators proposed in \[35，36\] use this 优化 technique to

processing， which could be of special interest， especially in embed-
increase 卷积神经网络 processing eﬃciency。

ded， 边-computing applications。

Each of the three presented “zero-skipping” techniques lead

To further improve energy and compute eﬃciency of 卷积神经网络 net-

to an improvement in the 卷积神经网络 instance processing time， but it

work processing， statistics of 特征 map data being processed

is clear that the “All zero-skipping” technique leads to the most

by 卷积神经网络 network can also be explored to reduce the number of

signiﬁcant improvement， although it is the most diﬃcult to im-

required memory accesses， using 特征 map compression tech-

plement. Proposed CoNNA 结构 is designed speciﬁcally to

nique. Eﬃcient 特征 map compression is possible because of the

implement this third， “All zero-skipping” 优化 technique。

heavy use of the ReLU 激活 function within 卷积神经网络 layers. The

Please notice that the majority of the previously proposed 卷积神经网络

ReLU 激活 function introduces a signiﬁcant number of zeros

accelerators \[15–30\] have completely ignored this “zero-skipping”

in the intermediate 特征 maps， by rectifying all negative out-

优化 opportunity。

put 特征 map values to zero. The number of zeros in 特征

maps depends on the information content of input data being fed

3\. CoNNA 卷积神经网络 accelerator details to 卷积神经网络，
卷积神经网络 结构 that is being used to process input data，

as well as the underlying problem that the 卷积神经网络 network is
trained

The CoNNA 卷积神经网络 accelerator is speciﬁcally designed to ex- to
solve. Nevertheless， it tends to increase as we move deeper into

ploit the “All zero-skipping” 优化 opportunity， described in the
卷积神经网络 network. For example， in VGG-16 卷积神经网络 almost 48% of
the

Section 2 . The CoNNA 结构 is designed to accelerate com- IFM values of
the CONV1_2 层 are zeros on average， and this

pressed pruned CNNs， created using some of existing ﬁne-grained value
goes up to 88% for the CONV5_3 层， Chen et al. \[19\] .

卷积神经网络 pruning algorithms， for example \[38–45\] . As the result
of Zero Run-Length encoding \[19\] has again been proposed to ef-

the application of 卷积神经网络 pruning procedure， 卷积神经网络 kernel
and weight ﬁciently exploit this phenomenon and compress zero values
found

maps will be sparse， since a certain number of weights will be re- in
the 特征 maps， by replacing consecutive runs of zeros with

moved from the 卷积神经网络 during pruning step. Similarly， because of
− the single run-length value. Using ZRL encoding only adds 5% 10%

ReLU 激活 function use， a signiﬁcant number of intermedi- overhead to
the theoretical entropy limit \[19\] . For example， in the

ate 特征 map values will be zero also. CoNNA is able to detect case of
VGG-16 卷积神经网络， using ZRL encoding results in the reduction of

on-the-ﬂy all product terms that will result in the zero-valued out-
required 特征 map transfer size per input image， from 60 MB in

come and skip their calculation， therefore reducing 层 process- the
uncompressed case， to only 30 MB in compressed case， when

ing time signiﬁcantly. This is done when convolutional， pooling and
16-bit ﬁxed-point representation of 特征 map data is used \[19\] .

fully-connected layers are being processed by the CoNNA. Further- In the
typical usage of 特征 map compression， all 特征

more， CoNNA is able to process 特征 and kernel maps data in maps，
except input image， are stored in compressed format in op-

compressed form， removing the need for decompression step， thus
erating memory. Accelerator reads encoded IFMs from operating

further shortening processing time and reducing the size of on- memory，
decompresses them using an appropriate decoder， and

chip memories used for storing IFM and KM data， which is a sig- uses a
decompressed data stream within the accelerator to com-

niﬁcant improvement compared to existing solutions \[19\] . pute output
特征 map. Computed output 特征 maps are op-

tionally processed by the ReLU module， compressed by the ZRL en-

coder， and transmitted to external memory. This saves both space 3. 1.
Principle of operation of the CoNNA 卷积神经网络 accelerator

and R/W bandwidth of the external memory. For example， this ap-

proach is used in MIT’s Eyeriss accelerator \[19\] . More than 90% of
the operations in 卷积神经网络 processing involve

However， please notice that a further improvement in the 卷积神经网络
convolutions \[3，10，51，52\] . Therefore， optimizing the execution of

processing time is possible if all MAC operations where IFM points
convolutional layers has an overwhelming impact on the overall

are zero-valued are “skipped” since they once more don’t change eﬃciency
and performance of any 卷积神经网络 accelerator. The process

the ﬁnal outcome of convolution or weighted sum calculations， of
transforming an input 特征 map into an output 特征 map

similar to “skipping” MAC operations with zero-valued convolu- by a
convolutional 层 can be described by Algorithm 1 . Please

tional coeﬃcients or fully-connected weight values， described pre- note
that， for the reason of simplicity， in presented pseudo-code

viously. adding of bias term for each convolutional kernel and optional
IFM

In principle， having a signiﬁcant number of zero values in in- padding
（necessary if we want to keep the horizontal and vertical

put 特征 maps， as well as in convolutional kernel and fully- size of
OFM equal to that of IFM） were omitted。

connected weight maps can be used to improve 卷积神经网络 instance pro-
A convolutional 层 takes as an input a 特征 map which

cessing eﬃciency by using one of three possible “zero-skipping” can be
represented as a 3D tensor of dimension IFM_Width x

× techniques： IFM_Height IFM_Depth . The convolutional 层 then
transforms

this input 特征 map， using Kernel_Num different convolutional

× 1. IFM “zero-skipping” - in this approach all MAC operations kernels，
into an output 特征 map of dimension OFM_Width

× where IFM point value is zero are skipped during the process of
OFM_Height Kernel_Num . Each convolutional kernel K is a 3D

× × convolutional and weighted sum calculation. For example，
卷积神经网络 tensor of dimension Kernel_Width Kernel_Height Kernel_Depth
，

<div class="page-break">

</div>

R. J. R. Struharik， B. Z. Vukobratovi ´c and A. M. Erdeljan et al. /
Microprocessors and Microsystems 73 （2020） 102991 7

tion of individual convolutions CoNNA 结构 processes IFM

Algorithm 1 Pseudo-code of generic convolutional 层 processing
algorithm。

bundles in the “Stick First” manner. Each IFM bundle that should

= \< ++ L1： for （ y 0； y OFM_Height； y ）

be convolved is divided into a number of IFM sticks， using nota-

= \< ++ L2： for （ x 0； x OFM_Width； x ）

tion from Fig. 1 ， and then each individual IFM stick is multiplied

= \< ++ L3： for （kn 0； kn Kernel_Num； kn ）

with the corresponding convolutional coeﬃcients from appropriate = \< ++
L4： for （kd 0； kd Kernel_Depth； kd ）

= \< ++ convolutional kernel stick. Furthermore， each individual
convolu- L5： for （kw 0； kw Kernel_Width； kw ）

= \< ++ L6： for （kh 0； kh Kernel_Height； kh ） tion is calculated
sequentially， meaning that loops L4-L6 are kept

+= OFM\[ x \]\[y\]\[kn\]

rolled. This is in sharp contrast with most of the previously pro-

∗ ∗ ∗ + + IFM\[ x S h kw \]\[y Sv kh\]\[ kd \] KM\[ kn \]\[kw\]\[ kh
\]\[kd\]；

posed 卷积神经网络 accelerators that try to increase processing speed by

partially or totally unrolling some or all of loops L4-L6。

where Kernel_Depth 参数 must be equal to the IFM_Depth

## Sequential approach of calculating individual convolutions em-

参数. Since Kernel_Width and Kernel_Height parameters are

ployed by the CoNNA 结构 is highly ﬂexible regarding the

usually much smaller than IFM_Width and IFM_Height parameters，

parameters of convolutions （kernel size， horizontal and vertical

each convolutional kernel is actually convolved many times， each

stride， kernel shape， etc. ）， resulting in highly-conﬁgurable accel-

time with different IFM bundle， using notation introduced in Fig. 1 .

erator， almost to the level of the accelerators based on CPUs and

## The way how kernels traverse IFM is controlled by two additional

GPUs， while still having high utilization of PB units， equal or even

parameters， horizontal and vertical kernel stride values， Sh and Sv .

higher than the MAC utilization of the accelerators based on the

Input 特征 map horizontal and vertical dimensions， stride pa-

custom hardware solutions. This high ﬂexibility is in sharp con-

rameters， together with horizontal and vertical padding values de-

trast with most of previously proposed 卷积神经网络 accelerators， which

termine the horizontal and vertical dimensions of resulting output

usually support only certain kernel conﬁgurations， allowing for ex-

特征 map， OFM_Width， and OFM_Height respectively. The depth

× × ample only 3 3 or 5 5 kernels with the stride of 1. Even if

of the resulting output 特征 map is unrelated to the depth of

they do support arbitrary kernel size and stride values， their eﬃ-

the input 特征 map and is purely determined by the number of

ciency drops signiﬁcantly when these parameters are set to some

different convolutional kernels that are deﬁned within the current

non-standard values。

convolutional 层， Kernel_Num .

Since individual convolutions are calculated sequentially， one

As can be seen from the pseudo-code of Algorithm 1 ， the pro-

MAC operation at a time， CoNNA 结构 can easily support

cess of computing the output from a convolutional 层 requires

convolutional kernels of any size， with different horizontal and

going through six nested loops. Please notice that there can be an

vertical sizes， with different horizontal and vertical strides， with-

additional 7th， outermost loop if batch processing is used， but for

out any degradation of the processing eﬃciency. Furthermore， the

simplicity reasons， it is not present in Algorithm 1 . At the core of

shape of the convolutional kernel doesn’t have to be squared， it

these nested loops are the MAC operations， where selected IFM

can be rectangular also， and it can even be triangular， oval， or

and KM points are multiplied and then added to the running sum

any other shape for that matter， including asymmetric shapes also。

of appropriate OFM point. This six nested loops algorithmic de-

Since each convolutional kernel is processed sequentially， one co-

scription of convolutional 层 operation creates a rather large de-

eﬃcient at a time， its actual shape and size are of no relevance to

sign space of possible computing architectures， each of them im-

the CoNNA 结构， and it also doesn’t affect the eﬃciency of

plementing a different type of parallelism， sequencing computa-

computing individual convolution。

tions and partitioning large IFM and KM data into smaller blocks

The second modiﬁcation of Algorithm 1 that is present in

that can then more easily ﬁt into smaller on-chip memories. It is

Algorithm 2 is the partial unrolling of loop L3， with the partial un-

worth noting that each of the previously proposed 卷积神经网络 hardware

rolling factor equalling the number of available Processing Block

accelerator architectures is actually representing one of these de-

modules， Num_PB . This 参数 is actually one of the conﬁg-

sign space points， CoNNA 结构 included。

uration parameters of the CoNNA 结构 and can be speci-

The CoNNA 结构 is based on the idea of sequen-

ﬁed at compile time to generate an instance of the CoNNA archi-

tial computation of individual convolutions from a convolutional

tecture with a speciﬁed number of Processing Blocks. This con-

层， using a single Processing Block （PB） unit for each convo-

ﬁgurability enables the scalability of the CoNNA 结构 in

lution that needs to be computed， and employing a number of

terms of achievable instance processing performance. As can be

PB units to compute several different convolutions in parallel。

seen from Algorithm 2 ， CoNNA uses existing Processing Block mod-

Algorithm 2 presents the pseudo-code of the convolutional 层

ules to concurrently compute a number of individual convolutions。

processing algorithm implemented in the CoNNA 结构。

## All these convolutions involve the same IFM bundle but use differ-

ent convolutional kernels。

Algorithm 2

Partial unrolling of loop L3 actually is one way of speeding-up
Pseudo-code of convolutional 层 processing algorithm， Implemented by
the

CoNNA 结构. the 卷积神经网络 processing used within the CoNNA 结构， by
paral-

lelizing the process of computation of individual convolutions. An- = \<
++

L1： for （ y 0； y OFM_Height； y ）

= \< ++ other way of speeding-up 卷积神经网络 processing used in CoNNA
is skip- L2： for （ x 0； x OFM_Width； x ）

= \< += L3： for （kn 0； kn Kernel_Num； kn Num_PB） ping ineffectual
MAC operations， by skipping all product terms that

= \< ++ L5： for （kw 0； kw Kernel_Width； kw ）

result in the zero-valued outcome. As previously stated， the CoNNA

= \< ++ L6： for （kh 0； kh Kernel_Height； kh ）

结构 implements “All zero-skipping” technique， presented

= \< ++ L4： for （kd 0； kd Kernel_Depth； kd ）

in Section 2 ， to skip all unnecessary multiplications， which will +=
{OFM\[ x \]\[y\]\[kn\]

∗ + ∗ + ∗ IFM\[ x S h kw \]\[y Sv kh\]\[ kd \] KM\[ kn \]\[kw\]\[ kh
\]\[kd\]； result in zero-valued outcome during convolutional， pooling
and

\+ += OFM\[ x \]\[y\]\[kn 1\] fully-connected 层 processing， so the
actual convolutional 层

∗ ∗ ∗ + + + IFM\[ x S h kw \]\[y Sv kh\]\[ kd \] KM\[ kn 1\]\[kw\]\[ kh
\]\[kd\]； …

processing algorithm that is implemented in the CoNNA architec-

\+ += OFM\[ x \]\[y\]\[kn Num_PB-1\]

ture is slightly different from Algorithm 2 . Algorithm 3 presents ∗ ∗
∗ + + + IFM\[ x S h kw \]\[y Sv kh\]\[ kd \] KM\[ kn N um \_PB-

the pseudo-code of the ﬁnal convolutional 层 processing algo-
1\]\[kw\]\[ kh \]\[kd\]； }

rithm， using the “All zero-skipping” technique， which is actually

implemented in the CoNNA 结构. Comparing Algorithm 2 with Algorithm 1 it
can be seen that

As opposed to Algorithms 1–3 operates on a compressed input two
modiﬁcations have been made. The ﬁrst modiﬁcation makes

特征 map， represented by CIFM tensor， and compressed convolu- loop L4
the innermost loop， which means that during the calcula-

<div class="page-break">

</div>

## 8 R. J. R. Struharik， B. Z. Vukobratovi ´c and A. M. Erdeljan et al. / Microprocessors and Microsystems 73 （2020） 102991

tional kernels， represented by CKM tensor. Compressed input fea- In
case of Algorithm 3 ， the number of MAC operations is not

ture map is the result of compressing zero-valued activations that
constant， since it depends on the number of non-zero valued prod-

are present due to the use of the ReLU 激活 function， while uct terms
that are present， which in turn depends on the num-

compressed convolutional kernels are the result of the 卷积神经网络
prun- ber of non-zero values in input 特征 map bundle and associated

ing process， described in Section 2 . These compressed structures
convolutional kernel， as well as on their actual positions within

contain only non-zero input 特征 map and kernel values， to-
uncompressed structures. However， on average， the number of re-

gether with the information about their positions in the original，
quired MAC operations to compute one convolution operation us-

uncompressed structures. ing “All zero-skipping” technique equals

Algorithm 3 implements the “All zero-skipping” tech-

= · · Num \_ MACs \_ Al g K ernel \_ W idth K ernel \_ Height K ernel \_
Depth

## 3 nique within the W1 loop， which replaces loops L4-L6 from

· · P \_ IF MNZ P \_ KMNZ （2）

Algorithms 1 and 2 . W1 loop is actually a while loop that is active

until all Processing Blocks have ﬁnished computing all non-zero

where P_IFMNZ is the percentage of non-zero valued points in the

product terms from their associated convolution. When all PBs

input 特征 map bundle， and P_KMNZ is the percentage of non-

complete their convolution calculation tasks， PBs_Busy（） function

zero valued convolutional coeﬃcients in the convolutional kernel。

will return a false value， indicating that the process of computing

## Theoretical speedup in computing one convolution using the

the next batch of Num_PB convolutions is completed and the

“All zero-skipping” technique over the “No zero-skipping” ap-

process of computing the next batch of Num_PB convolutions can

proach equals

commence。

Num \_ MACs \_ Al g Each PB actually implements Calc_Next_NZPT（）
function. This /

1 2 = v Con olution \_ Cal cul ation \_ Speedup

Num \_ MACs \_ Al g function determines on-the-ﬂy， on a clock by clock
basis， the next

3

non-zero valued product term that should be computed， based 1

= （3）

· · + on supplied compressed IFM bundle， CIFM\[x Sh： x Sh
Kernel_Width- ·

P \_ IF MNZ P \_ KMNZ

· · + 1\]\[y Sv： y Sv Kernel_Height-1\]\[： \]， and compressed
convolutional ker-

## Convolution calculation speedup clearly depends on the num-

nel， CKM\[kn\]\[： \]\[： \]\[： \] . If there are no more non-zero
product terms

ber of zeros that are present in the input 特征 map bundle and

Calc_Next_NZPT（） function returns zero value， which is an indica-

convolutional kernel. While the number of zeros in the input fea-

tion that the PB has ﬁnished computing the convolution operation

ture maps cannot be directly controlled and also depends on the

associated with it。

input instance that is currently being processed by the 卷积神经网络
net-

## Please notice that the number of non-zero valued product

work， the number of zeros in the convolutional kernel depends on

terms present in individual convolution operation depends on

the pruning ratio that was achieved during 卷积神经网络 network pruning

the number of non-zero values in the input 特征 map bun-

process. Using previously published research we can estimate the

dle and associated convolutional kernel. Since all PBs operate on

typical number of zeros in the input 特征 maps， which ranges

the same input 特征 map bundle， the number of non-zero IFM

from 25% to 88% \[19\] ， and the typical number of zeros in con-

points is identical for all PBs. However， the number of non-zero

volutional kernels， which ranges from 16% to 78% \[43\] . Based on

convolutional coeﬃcients can vary from one convolutional ker-

these ﬁgures， the achievable speedup of computing individual con-

nel to another. If this would be allowed， computing eﬃciency of

volutional layers using the “All zero-skipping” technique over the

Algorithm 3 could be severely degraded， because in this case， the

“No zero-skipping” approach ranges from 1. 59 up to 37. 88， which

number of non-zero product terms each PB must compute could

is clearly a signiﬁcant improvement。

vary signiﬁcantly. In this scenario， all PBs that have already ﬁn-

Computing output of a fully-connected 层 is the second

ished with computing their associated convolution would have to

most computationally demanding 卷积神经网络 层 type， after the con-

wait for the slowest performing PB to complete its calculation pro-

volutional 层. All other standard 层 types， including pool-

cess. In order to limit this variation， we can enforce that each con-

ing， adding， concatenation， have a negligible computational load，

volutional kernel from a given convolutional 层 has to have an

compared to convolutional and fully-connected layers. Please no-

identical number of non-zero valued coeﬃcients after pruning is

tice that the computation of a fully-connected 层 can be

performed. In other words， during pruning identical number of

mapped to a convolutional 层 computation in the following

convolutional coeﬃcients will be removed in every convolutional

way. Each neuron from the fully-connected 层 computes the

kernel from a given convolutional 层. Experiments that authors

weighted sum over the entire input 特征 map， using its own

have conducted seem to indicate that this constraint is not signiﬁ-

speciﬁc set of weights. However， this weighted sum computa-

cantly limiting the previously reported maximum achievable prun-

tion can be seen as the special case of convolutional sum com-

ing rate for all 卷积神经网络 networks used in \[ 3 ， 10 ， 52 ，53\].

putation， where the convolution involves the entire input fea-

## Please notice that even having convolutional kernels with an

ture map and convolutional coeﬃcients equal fully-connected neu-

identical number of non-zero coeﬃcients will not result in iden-

ron weight values. Therefore， computing fully-connected 层 of

tical convolution computation times for different PBs. This is be-

## N neurons is equivalent to computing N different convolutions

cause the distribution of these non-zero coeﬃcients will vary be-

spanning the entire input 特征 map. The result of processing

tween different convolutional kernels， and when these kernels

a fully-connected 层 will be an output 特征 map of dimen-

are correlated with the IFM bundle they could still have an un-

× sion 1 1x N . This means that we could use Algorithms 1 – 3

equal number of non-zero valued product terms to compute. In

to compute the output of a fully-connected 层 also， conﬁgur-

Section 4. 4 a detailed analysis of this effect on the overall
卷积神经网络

= ing algorithm parameters in the following way： OFM_Width 1；

processing eﬃciency of the CoNNA 结构 will be presented。

= = = OFM_Height 1； Kernel_Num N； Kernel_Width IFM_Width； Ker-

## Let us compare the number of MAC operations required to

= = nel_Height IFM_Height； Kernel_Depth IFM_Depth .

compute one convolution using Algorithms 1–3 . In the case of

Algorithms 1 and 2 ， the number of operations is constant and

3\. 2. Overview of the CoNNA 卷积神经网络 accelerator 结构

equals

The CoNNA 结构 is designed to act as a standalone 卷积神经网络

= · · hardware accelerator. It is designed as a standard soft-IP core，
that Num \_ MACs \_ Al g K ernel \_ W idth K ernel \_ Height K ernel \_
Depth

/ 1 2

can be easily integrated into contemporary System-on-Chip （SoC）

（1）

<div class="page-break">

</div>

R. J. R. Struharik， B. Z. Vukobratovi ´c and A. M. Erdeljan et al. /
Microprocessors and Microsystems 73 （2020） 102991 9

Fig. 3. CoNNA Compressed 卷积神经网络 Accelerator： a） Integration into
SoC or PSoC； b） Top level 结构 with available interfaces。

• or Programmable System-on-Chip （PSoC） solutions. CoNNA uses Output
Stream Manager （OSM） – used to format， compress and

standard AXI interfaces， so it can be easily connected with ARM or
stream output 特征 map data， generated by the RCU as it

RISC-V microprocessors， using standard AXI-Interconnect modules，
processes 卷积神经网络 layers， to the external DRAM memory or on-chip

as shown in Fig. 3 a. cache memory。

• Having a microprocessor within the system eases the control
Conﬁguration and Control Unit （CCU） – used to control the op-

and conﬁguration of the CoNNA 卷积神经网络 accelerator， but it must be
eration of the CoNNA 卷积神经网络 accelerator and enable interfacing

stated that conﬁguration and control of the CoNNA accelerator is a with
surrounding logic。

relatively simple process， so a pure hardware-based solution， with

The CoNNA 卷积神经网络 accelerator uses four AXI interfaces to commu-
hardwired FSM controlling CoNNA， can also be used. However， in

nicate with surrounding on-chip modules： order for CoNNA to be fully
functional， some kind of memory

needs to be present in the system， which could be on-chip cache

• Input Stream Interface （ISI） – AXI-Full interface used to stream

memory or off-chip operating memory， as shown in Fig. 3 a. If the

input data， including input image， intermediate input 特征

system can implement large enough on-chip cache memory， as

maps and kernel maps for selected 卷积神经网络 network， to the CoNNA

shown in Fig. 3 a， of an order of several tens of megabytes， which is

accelerator module for processing. CoNNA acts as the master of

today possible even if we use FPGA devices， the result would be a

the ISI interface. The other side of the ISI interface is connected

highly power-eﬃcient solution. In this conﬁguration， all 卷积神经网络
topol-

either to the on-chip cache module or to the DRAM memory

ogy data， as well as all intermediate 特征 maps would be stored

controller module。

inside the on-chip cache. This would signiﬁcantly reduce required

• Output Stream Interface （OSI） – AXI-Full interface used to

data movement between external memory and the CoNNA acceler-

stream output 特征 maps data to the on-chip cache or exter-

ator and consequently lower power consumption since most of the

nal DRAM memory. CoNNA acts as the master of the OSI inter-

energy in data processing systems is spent in moving data between

face. The other side of the OSI interface is connected either to

external DRAM memory and processing system \[47\] . However， for

the on-chip cache module or to the DRAM memory controller

low-cost solutions， this conﬁguration will probably not be possi-

module。

ble， so in this case， instead of using an on-chip cache， the system

• 卷积神经网络 Description Interface （CDI） – AXI-Full interface used
to

would have to include external operating memory， usually imple-

load structural information about the 卷积神经网络 network that is be-

mented as DRAM memory， as shown in the Fig. 3 a。

ing accelerated. CDI interface is implemented using standard

The top-level 结构 of the CoNNA compressed 卷积神经网络 ac-

AXI-Full interface protocol. CoNNA acts as the master of the

celerator is shown in the Fig. 3 b. The CoNNA 卷积神经网络 accelerator
is

CDI interface. The other side of the CDI interface is connected

composed of the following four modules：

either to the on-chip cache module or to the DRAM memory

controller module。

• Reconﬁgurable Computing Unit （RCU） – used to perform all •

Conﬁguration Interface （CI） – AXI-Lite interface used to conﬁg-

computations deﬁned by different layers from the 卷积神经网络 net- ure
and control the operation of the CoNNA 卷积神经网络 accelerator. CI

work （including convolutional， depthwise convolutional， pool-
interface is implemented using standard AXI-Lite interface pro-

ing， concatenation， adding and fully connected layers）， exploit-
tocol. CoNNA acts as the slave of the CI interface. The other side

ing the sparsity of kernel and 特征 maps. RCU module is fur- of the CI
interface is connected either directly to the host pro-

ther composed out of a number of Processing Block modules. cessor or
hardwired FSM or to some AXI interconnect module。

• Input Stream Manager （ISM） – used to supply all input data

（conﬁguration， kernel map， input 特征 map）， in compressed
Furthermore， the CoNNA accelerator has an interrupt request

format， coming from external DRAM memory or on-chip cache output port，
which is used to signal to the surrounding mod-

memory to the appropriate internal modules of the RCU mod- ules that
CoNNA has ﬁnished processing one input instance， and

ule. that the instance classiﬁcation data is available. Since the CoNNA

<div class="page-break">

</div>

## 10 R. J. R. Struharik， B. Z. Vukobratovi ´c and A. M. Erdeljan et al. / Microprocessors and Microsystems 73 （2020） 102991

Fig. 4. Linked list representation of target 卷积神经网络 structural
description that will be accelerated by the CoNNA 卷积神经网络
accelerator。

• 卷积神经网络 accelerator uses AXI interfaces， integration within
ARM/RISC- Classiﬁcation Data Buffer （CDB） – memory buffer used for
stor-

V based SoCs or PSoCs is greatly simpliﬁed. All data that is being ing
classiﬁcation data for the last input instance that was pro-

processed or generated by the CoNNA accelerator is stored either cessed
by the CoNNA 卷积神经网络 accelerator. The output 特征 map

in the internal on-chip cache memory or external DRAM memory， generated
by the ﬁnal 层 of the target 卷积神经网络 network is stored

depending on the available memory resources in the system. in this
buffer. This data represents the classiﬁcation information

The operation of the CoNNA 卷积神经网络 accelerator is conﬁgured and
about the instance that was processed by the 卷积神经网络 network， and

controlled using a set of internal registers， as well as several data
it can be used by higher application levels within the complete

structures required for correct operation， which are stored either in
system。

the on-chip cache memory or external DRAM memory. The CoNNA

卷积神经网络 accelerator operates on the following data structures：
CNNSD linked list， shown in the Fig. 4 ， deﬁnes the topology，

together with various 参数 values， of the target 卷积神经网络 network

• 卷积神经网络 Structural Description （CNNSD） list – this is a linked
list that should be accelerated by the CoNNA 卷积神经网络 accelerator.
CNNSD

of nodes， each one describing one 层 from a particular 卷积神经网络
list is composed of a number of nodes， the exact number being

network that the CoNNA accelerator should process. equal to the number
of layers in the target 卷积神经网络 network， where

• Input Instance Buffer （IIBs） – memory buffer， used for storing each
节点 describes the characteristics of one particular 层 from

input instances that should be processed by the CoNNA 卷积神经网络 the
selected 卷积神经网络 network。

accelerator. Every CNNSD 节点， shown in Fig. 4 ， contains the
following

• 特征 Map Buffers （FMBs） – a number of memory buffers， ﬁelds：

used for storing input 特征 maps and output 特征 maps

• data that are processed by various layers from the target 卷积神经网络
层 Info – this ﬁeld， which has multiple subﬁelds， is used to

network. The minimum number of FMB buffers equals two， one specify
details about the current 卷积神经网络 层 that should be pro-

for storing input 特征 map that is being processed by the cessed by
CoNNA. 层 Info ﬁeld is composed of the following

current 卷积神经网络 层 and another for storing output 特征 map
subﬁelds：

◦ that is being generated by the current 卷积神经网络 层. This conﬁg- 层
Type – this ﬁeld speciﬁes the type of the current

uration is suﬃcient in case the target 卷积神经网络 network topology
层， valid values being： convolutional， depthwise convolu-

doesn’t contain adding or concatenation layers， like in the case
tional， pooling （average or max pooling）， adding， concatena-

of AlexNet， VGG or MobileNet V1 卷积神经网络 architectures. However，
tion， and fully-connected。

◦ if the target 卷积神经网络 network topology contains adding or
concate- IFM Width – this ﬁeld speciﬁes the width of the input fea-

nation layers， which is， for example， the case with ResNet， In- ture
map， expressed as the number of IFM points。

◦ ception and NASNet architectures， then additional FMB buffers IFM
Height – this ﬁeld speciﬁes the height of the input fea-

are necessary. ture map， expressed as the number of IFM points。

<div class="page-break">

</div>

R. J. R. Struharik， B. Z. Vukobratovi ´c and A. M. Erdeljan et al. /
Microprocessors and Microsystems 73 （2020） 102991 11

◦ IFM Depth – this ﬁeld speciﬁes the depth of the input fea- erated is
possible when using the CoNNA 卷积神经网络 accelerator， with-

ture map， expressed as the number of IFM points. out the need to modify
and re-implement the accelerator itself。

◦ Number of Kernels – this ﬁeld speciﬁes the number of ker- This
situation is shown in the Fig. 5 . There can be a number

nels used to process IFM data by the current convolutional of
卷积神经网络 topology description lists， sitting in the operating mem-

层. This number also determines the depth of the OFM ory to which the
CoNNA accelerator is connected. By simply writ-

generated by the current convolutional 层. This ﬁeld is ing the correct
base address of the target 卷积神经网络 description list in

valid only if the “层 Type” ﬁeld is set to “convolutional” the CNNSD
Pointer register， the user can easily select the desired

or “depthwise convolutional”. 卷积神经网络 network that should be
processed by the CoNNA accelera-

◦ Kernel Size – this ﬁeld speciﬁes the horizontal or vertical tor. This
selection can be done on-the-ﬂy， without any need for

size of the square convolutional kernel， or pooling area， ex- hardware
reprogramming or reconﬁguration. Switching between

pressed as the number of KM points. This ﬁeld is valid only different
卷积神经网络 networks that should be accelerated is extremely

if the “层 Type” ﬁeld is set to “convolutional”， “depthwise fast，
requiring only one write to CoNNA’s internal register （CNNSD

convolutional”， “max pooling” and “average pooling”. Pointer）. This
特征 of the CoNNA accelerator is of great impor-

◦ Kernel Stride – this ﬁeld speciﬁes the stride of the convolu- tance in
the case of applications where rapid， dynamic switching

tional kernel or pooling area. This ﬁeld is valid only if the of
卷积神经网络 networks is necessary. It is worth noting that many previ-

“层 Type” ﬁeld is set to “convolutional”， “depthwise con- ously
proposed 卷积神经网络 accelerators， particularly accelerators designed

volutional”， “max pooling” and “average pooling”. using HLS
techniques， cannot support this kind of functionality。

◦ Padding – this ﬁeld speciﬁes if padding should be used Furthermore，
on-the-ﬂy modiﬁcation of accelerated 卷积神经网络 net-

when processing IFM data with the current 卷积神经网络 层. work topology
is also possible when using CoNNA 卷积神经网络 accelerator，

◦ 激活 Function Type – this ﬁeld speciﬁes the type of removing or adding
卷积神经网络 layers or changing parameters of exist-

激活 function to be used， valid values being： “none”， ing layers， by
a simple modiﬁcation of the target 卷积神经网络 description

“ReLU” or “arbitrary”. In the case of the “arbitrary” type， linked
list. For example， in the Fig. 5 卷积神经网络 description linked list

th additional 激活 Function Lookup Table Pointer ﬁeld for M 卷积神经网络
network could be modiﬁed by replacing the existing

points to the beginning of the 激活 function lookup description of 层 2
with the description of some other 层， for

∗ table content memory block. This block is loaded into the example， 层
2 . This can easily be done， by simply rewriting ex-

CoNNA’s Arbitrary Non-Linear 激活 Function Calculator isting values of
the “Next 层 Pointer” ﬁeld from 层 1 节点

module， located within the RCU module， to specify the ex- and IFM Data
Pointer ﬁeld from 层 3 节点. After these modiﬁca-

th act shape of the non-linear 激活 function that should tions， the
topology of M 卷积神经网络 network will be modiﬁed to include

∗ be used within the current 卷积神经网络 层. In the case of “none” a
new 层， 层 2 ， instead of existing 层 2. As can be seen，

and “ReLU” values， 激活 Function Lookup Table Pointer this modiﬁcation
is very quick， requiring only a couple of mem-

is ignored. ory accesses to the operating memory where the 卷积神经网络
description

• Input 特征 Map Data Pointer – this ﬁeld speciﬁes the base list is
stored. Therefore， rapid， dynamic modiﬁcation of target 卷积神经网络

address of the memory block， which holds the values of the topology is
also supported by the CoNNA 卷积神经网络 accelerator， which

input 特征 map of the current 卷积神经网络 层. can be of particular
interest when working with hierarchical CNNs

• Output 特征 Map Data Pointer – this ﬁeld speciﬁes the base \[48–50\] .

address of the memory block， which will store the values of the CoNNA
processes the 卷积神经网络 network in a sequential manner， one

output 特征 map， generated by the current 卷积神经网络 层. 层 at a
time. It uses information about the current 卷积神经网络 层

• Kernel Data Pointer – this ﬁeld speciﬁes the base address of the to be
accelerated， stored in the appropriate 节点 of the associ-

memory block， holding the kernel coeﬃcients， in case of a con- ated
卷积神经网络 description linked list， to reconﬁgure the RCU unit and

volutional or depthwise convolutional 层， or weight values in perform
all necessary computations deﬁned within that 卷积神经网络 层。

case of a fully connected 层. When all computations deﬁned in the
current 层 are completed，

• Next 层 Pointer – this ﬁeld speciﬁes the address of the next output
特征 map data generated by the current 层 is stored

CNNSD 节点， which describes the characteristics of the subse- in the
appropriate memory buffer and CoNNA can proceed with

quent 层 from the 卷积神经网络 network that is being accelerated. If the
next 卷积神经网络 层 from the supplied CNNDS linked list. This op-

the value of this ﬁeld is NULL ， this is an indication that the eration
is repeated until a ﬁnal 节点 from the CNNDS linked list

current 卷积神经网络 层 is actually the ﬁnal 层 of the 卷积神经网络 and
is processed， indicating the end of processing of the current input

that the CoNNA accelerator should signal completion of the cur- instance
by selected 卷积神经网络. The process of sequential processing of

rent input instance classiﬁcation， after it has processed this ﬁ-
卷积神经网络 layers employed by the CoNNA 卷积神经网络 accelerator is
shown in

nal 层. the Fig. 6 .

Please note that although CoNNA processes 卷积神经网络 layers sequen-

Besides CNNSD linked list， IIB， FMB and CDB buffers， the CoNNA

tially， 层 by 层， computations deﬁned within each 卷积神经网络 层

卷积神经网络 accelerator has the following set of internal registers，
that are

are executed concurrently， using available Processing Block mod-

used to conﬁgure and control its operation. All registers are 32-bit，

ules from the RCU module. The way how this is done was pre-

and are accessed using the CI AXI-Lite interface of the CoNNA
卷积神经网络

sented at the beginning of Section 3 ， and will also be explained in

accelerator：

more detail in Section 3. 2. 1 .

• Control Register – used to control the operation of the CoNNA

卷积神经网络 accelerator。

3\. 2. 1. Details of RCU module

• Status Register – used to monitor the operation of the CoNNA

RCU module is the central module of the CoNNA 卷积神经网络 accel-

卷积神经网络 accelerator。

erator. It is used to perform all necessary numerical calculations，

• CNNSD Pointer – register holding the base address of the

deﬁned by different 卷积神经网络 layers. RCU module， shown in Fig. 7 ，
is

CNNSD linked list， where the ﬁrst CNNSD 节点 is located. The

designed as a coarse-grained reconﬁgurable hardware module， al-

CoNNA 卷积神经网络 accelerator uses this address value to fetch the in-

lowing easy， fast， dynamic， on-the-ﬂy reconﬁguration， in order to

formation about the structure of the ﬁrst and all subsequent

create different dataﬂows， optimized for processing particular
卷积神经网络

layers of the 卷积神经网络 network that is being accelerated。

层 type。

Because 卷积神经网络’s structural information is stored in a form of a
RCU module is composed of a number of Processing Block mod-

linked list， on-the-ﬂy selection of target 卷积神经网络 network to be
accel- ules and one 激活 Function Calculator （AFC） module. PBs are

<div class="page-break">

</div>

## 12 R. J. R. Struharik， B. Z. Vukobratovi ´c and A. M. Erdeljan et al. / Microprocessors and Microsystems 73 （2020） 102991

![](/workspace/CoNNa_zh_media/b567810f02c2ebb003f7dfe8d3c654be12472835.png)

Fig. 5. Principle of on-line switching between different CNNs and
accelerating CNNs with dynamic topology。

![](/workspace/CoNNa_zh_media/3dda74979f9f59aff09674718c44c2c8d1e31b50.jpg)

Fig. 6. Principle of sequential 层 processing during 卷积神经网络
acceleration implemented by CoNNA。

able to perform all required 卷积神经网络 层 processing operations，
using In the Fig. 7 b distribution of different convolutions over
available

compressed input 特征 map and kernel map data， implement- PBs， in case
of processing convolutional 层 with 6 different con-

ing the “All zero-skipping” technique， to increase 卷积神经网络
computation volutional kernels， is presented. Please notice that if the
number

performance. AFC module is used for post-processing data com- of
convolutional kernels is smaller than the number of PBs， some

ing from PB modules. AFC module contains a number of Rectiﬁed of the PBs
will be inactive. Also note that the amount of paral-

Linear Units （ReLUs）， implementing the Rectiﬁed Linear 激活 lelism is
limited by the number of available PBs， since their num-

function and one Arbitrary Non-Linear 激活 Function Calcu- ber
determines the partial unrolling factor that will be used to un-

lator unit， capable of implementing arbitrary non-linear 激活 roll loop
L3 from Algorithm 3 ， Num_PB . However， for most stan-

functions， which are mainly used in fully-connected 卷积神经网络
layers. dard 卷积神经网络 networks \[ 3 ， 10 ， 52 ，53\] number of
different convolutional

As already mentioned in Section 3. 1 ， during convolutional 层 kernels
speciﬁed in convolutional layers rises sharply as we move

processing CoNNA allocates computation of one complete convolu- deeper
inside the 卷积神经网络 network， and quickly reach values above

tion to a single PB. Multiple convolutions， operating on the same 100，
so this constraint is usually not so severe。

IFM bundle are being calculated in parallel， by different PBs， as The
detailed 结构 of Processing Block is shown in the

shown in the Fig. 7 b， and previously described by Algorithm 3 . Fig. 8
. PB contains non-zero product term detection logic， imple-

<div class="page-break">

</div>

R. J. R. Struharik， B. Z. Vukobratovi ´c and A. M. Erdeljan et al. /
Microprocessors and Microsystems 73 （2020） 102991 13

Fig. 7. Details of the RCU Module： a） top-level 结构 of the RCU
Module； b） mapping individual convolutions to available PB units。

Fig. 8. The 结构 of the processing block module。

mented inside the Data Fetcher （DF） module， which allows the ex-
eﬃcients and their actual positions in the original uncompressed

ecution of numerical computations only when their result will be kernel.
Remaining two memories do the same for the IFM bun-

different from zero， skipping all unnecessary （zero-outcome） com- dle
that is currently being processed by the PB module. Please

putations， resulting in a signiﬁcant speedup of the 卷积神经网络
calcula- notice that all data stored in these four memories are stored
in

tion process. Please notice that the DF module actually implements a
serialized format， as the result of ﬂattening loops L4-L6 from

the Calc_Next_NZPT（） function from Algorithm 3 ， directly in hard-
Algorithm 2 ， using the “Stick First” approach described during the

ware. PB also contains four local memories， located in the Local
analysis of Algorithm 2 . Computing Unit （CU） performs all neces-

Memory module （LM）， used for storing data from selected convo- sary
operations on the selected kernel and IFM value pairs， sup-

lutional kernel and IFM bundle， in compressed format. Two mem- plied by
the DF module. It contains one MAC unit， together with

ories are used to store all non-zero valued convolution kernel co- some
additional logic， for enabling different compute dataﬂows，

<div class="page-break">

</div>

## 14 R. J. R. Struharik， B. Z. Vukobratovi ´c and A. M. Erdeljan et al. / Microprocessors and Microsystems 73 （2020） 102991

Fig. 9. Zero-valued product term skipping principle of operation。

• depending on the type of 卷积神经网络 层 that is being processed by
the Input Stick Buffer （ISB） memory – used for storing cached IFM

CoNNA 卷积神经网络 accelerator. Output FIFO is used to synchronize pro-
sticks in compressed format。

• cessing steps performed by different PBs since different PBs can Stick
Valid （SV） memory – used to store information is the IFM

require different periods of time to calculate their associated con-
stick located at the same position in ISB memory valid for read-

volutions. This can occur because of the possible difference in the ing
or not。

• actual number of non-zero valued product terms that are present Read
Controller （RC） – used to read selected compressed IFM

in different convolutions， as it was described during the presen- stick
from ISB memory and transfer it to the RCU module， more

tation of Algorithm 3 . Finally， the Conﬁguration Register is used
speciﬁcally into two local memories located inside the LM mod-

to specify the desired PB operating mode that should be used. ule of
each PB module。

• PB module supports several datapath conﬁgurations， which enable Write
Controller （WC） – used to write compressed IFM stick

eﬃcient processing of convolutional， pooling， fully-connected and from
external DRAM memory or on-chip cache memory to ap-

adding 卷积神经网络 层 types. propriate position inside ISB memory。

• The principle of the “All zero-skipping” technique used to skip Input
Stream Router （ISR） – used to select which input data

all zero-valued product terms， used by the PB module during the stream
will be sent to the RCU module. Two possible input data

convolution calculation process， is shown in the Fig. 9 . LM module
stream sources are available：

◦ streams information about the positions of kernel and IFM bun- First，
coming from the RC module， used when compressed

dle non-zero values to the DF module. DF module detects on-the- IFM
sticks are being transferred to the RCU。

◦ ﬂy the next kernel/IFM non-zero valued pair， implementing the
Second， coming directly from Input Stream Interface， used

“All zero-skipping” technique， using the information about the ker-
when convolutional kernel map or fully-connected weight

nel/IFM non-zero value positions in the uncompressed data struc- map
data is being transferred to the RCU module。

tures， and passes it to the CU module for processing. This is ac-

## The central module of ISM is the Input Stick Buffer memory complished by a parallel search for the next coincident non-zero

module. This memory is used for storing selected compressed IFM valued
position in both KM and IFM data streams， within a spec-

sticks from the current input 特征 map， which is being pro- iﬁed search
window， which is 16 elements wide in the case of

cessed by the current 卷积神经网络 层. During input 特征 map pro- the
CoNNA 结构. This operation is repeated until all data

cessing by the convolutional layers， IFM sticks are being repeatedly is
used， and all relevant， non-zero valued product terms are accu-

reused during the calculation of adjacent convolutions， as convo-
mulated within the CU module， to obtain the ﬁnal value of convo-

lutional kernels slide over input 特征 map. This will always be lution
calculation operation。

the case if the horizontal and vertical stride values， Sh and Sv ， are

3\. 2. 2. Details of ISM and OSM modules smaller than the corresponding
horizontal and vertical kernel sizes，

ISM module is used to stream input data to the RCU module. In-
Kernel_Width and Kernel_Height . This opens a possibility to mini-

put stream data consists of PB conﬁguration data， input image data mize
data movement between the 卷积神经网络 accelerator and external

or input 特征 map data， and convolutional kernel coeﬃcient val- memory
by caching IFM sticks that will be reused during the ad-

ues or fully-connected weight values， depending on the type of jacent
convolutions computation process. ISB memory module acts

卷积神经网络 层 that is currently being processed. All
卷积神经网络-related data， as this local cache of selected IFM sticks.
ISB stores selected com-

× which include all above-mentioned data except PB conﬁguration pressed
IFM sticks， 1 1xS sections of the IFM， where S is the

## D D

data， is received either from on-chip or external DRAM memory number of
non-zero valued IFM points that are present in the cur-

in compressed format， which is then either stored inside the Input rent
IFM stick， which will be used in the upcoming convolution

Stick Buffer （ISB） module， in case of the input 特征 map data， or
calculation operations， as shown in the Fig. 10 b. Please notice that

routed directly to the appropriate LM module inside the selected the 3D
cube from Fig. 10 b， which represents the content of the ISB

PB module， in case of convolutional coeﬃcients or fully-connected
memory module， is actually toothed since different IFM sticks can

weights. Please notice that the CoNNA 卷积神经网络 accelerator is
designed contain a different number of non-zero valued points。

to process all incoming 卷积神经网络-related data in compressed format，
so Each time the same IFM stick is needed in the convolution cal-

there is no need to decompress it. Fig. 10 presents details about the
culation operation， instead of re-fetching it from external DRAM

internal organization of the Input Stream Manager module， with all
memory， it is fetched from the ISB module， reducing the number

major sub-modules shown. of data transfers from external DRAM memory，
thus saving power。

## Input Stream Manager is composed of the following major mod- Once all convolution operations involving given IFM stick are com-

ules. pleted， that IFM stick can be removed from the ISB module and

<div class="page-break">

</div>

R. J. R. Struharik， B. Z. Vukobratovi ´c and A. M. Erdeljan et al. /
Microprocessors and Microsystems 73 （2020） 102991 15

Fig. 10. Input stream manager： a） Top level ism modules； b）
Organization of ISB and SVM memories。

Fig. 11. Principle of ISB memory operation。

the next IFM stick can be loaded in its place from external DRAM
Required ISB memory size clearly depends on the size of the

memory. input 特征 map that is being cached and the percentage of non-

Read Controller and Write Controller modules are charged with zero
valued IFM points， but is signiﬁcantly smaller than the mem-

correct manipulation of IFM sticks stored inside the ISB cache ory
required to store the complete IFM map， because Kernel_Height

memory module. WC module sweeps the input 特征 map line 参数 is always
signiﬁcantly smaller than the IFM_Height pa-

by line， as shown in Fig. 11 ， and writes IFM sticks in appropriate
rameter。

positions inside the ISB memory module. Please notice that the re- On
the other hand， the RC module reads selected IFM sticks

quired size of ISB memory is signiﬁcantly smaller than the size of from
ISB memory and transfers them to the RCU module. RC mod-

the complete input 特征 map that is being processed by the cur- ule
reads IFM sticks in a different order from the order WC module

rent convolutional 层. ISB memory size， measured in the num- uses to
write them in the ISB memory. While WC traverses input

ber of IFM points， that is suﬃcient to correctly cache IFM memory 特征
map horizontally， loading complete IFM line before moving

sticks from a given 卷积神经网络 network equals to the next one， RC
traverses ISB memory in the more localized

manner， deﬁned by the frontal shape of the convolutional kernel，

× as shown in the Fig. 11 in the case of 3 3 kernel。

{ } = ISB \_ Memory \_ Size max IF M \_ W idt h

## Please notice that the speed at which the WC module writes

i

∈ i layers

## IFM sticks into ISB memory can differ from the speed at which the

· max Kernel \_ Heigh t RC module reads IFM sticks from ISB memory. This
can happen be-

， i j

∈ i layers

cause data processing throughputs of the memory subsystem and ∈

j ker nels

RCU module can be different. Therefore， a mechanism that ensures

· max S （4）

D ， ， the consistency of ISB memory content needs to be devised. This
i j k

∈ i layers

≤ \< 0 j IF M \_ W id t h is the purpose of the Stick Valid memory
module. SV memory is

i ≤ \<

## 0 k IF M \_ Heigh t i

<div class="page-break">

</div>

## 16 R. J. R. Struharik， B. Z. Vukobratovi ´c and A. M. Erdeljan et al. / Microprocessors and Microsystems 73 （2020） 102991

· · IFM_Width Kernel_Height 1 bits large. For each IFM stick from ISB
Table 1

Characteristics of the CNNs used in comparison experiments. memory there
is a corresponding bit in the SV memory， indicat-

ing is that particular IFM stick valid or not. Each time WC module No.
of Layers （Conv， Pool，

wants to write a new IFM stick into selected location inside the ISB
卷积神经网络 结构 Add， Fully-Conn） No. of Parameters

memory it ﬁrst checks is the corresponding bit from the SV mem-
\[Millions\] No. of Operations

ory cleared， which indicates that all convolutions which use IFM
\[GOps\]

AlexNet \[3\] （16， 5， 0， 3） 60. 93 1. 45 stick currently stored at
the selected location inside ISB memory

VGG-16 \[10\] （16， 5， 0， 3） 138. 35 30. 95

have been computed， and it can be replaced with a new IFM stick。

VGG-19 \[10\] （19， 5， 0， 3） 143. 66 39. 28

If this is the case， WC writes a new IFM stick in the selected lo-

GoogleNet \[52\] （21， 5， 0， 1） 6. 97 3. 16

cation inside ISB memory and sets the corresponding bit inside SV
ResNet-18 \[53\] （17， 2， 8， 1） 11. 51 3. 59

memory. Otherwise， the WC module needs to wait until the corre-
ResNet-50 \[53\] （49， 2， 16， 1） 25. 50 7. 72

sponding SV memory bit is cleared. When the RC module wants to

read IFM stick from ISB memory it ﬁrst checks is the IFM stick at

celerator， originally developed by a startup company Deephi \[30\] ，

selected ISB memory location valid or not， by checking the corre-

now is the part of the IP portfolio of the largest FPGA manufacturer

sponding bit from the SV memory. If the bit is set， this means that

Xilinx， who recently bought DeePhi. Finally， NVLDA \[51\] is one

the IFM stick is valid and RC can transfer it to the RCU module，

of the ﬁrst open-source based 卷积神经网络 accelerators coming from the

otherwise the RC module has to wait until the WC module certi-

industrial sector， developed by NVIDIA， the current leader in the

ﬁes that the IFM stick data is valid。

deep learning acceleration ﬁeld. The performance of the CoNNA

The operation of ISB memory is shown in the Fig. 11 ， which

卷积神经网络 accelerator was compared with seven previously proposed

× shows a snapshot of the input 特征 map， 6 6 IFM sticks in

卷积神经网络 accelerators using six well-known 卷积神经网络 networks，
AlexNet \[3\] ，

size. Please notice that there is a third dimension to the IFM since

VGG-16， VGG-19 \[10\] ， GoogleNet \[52\] ， ResNet-18 and ResNet-50

its depth is always bigger than one， but for clarity， it was omitted。

\[53\]. Relevant data for each 卷积神经网络 network used in experiments
are

× If we assume that this IFM is being processed by a 3 3 kernel

presented in Table 1 .

with horizontal and vertical strides of 1， then each IFM stick will

Conﬁgurations of all reference 卷积神经网络 accelerators that were used

be used in the process of calculating of up to nine different convo-

in performance comparison experiments are presented in Table 2 .

lutions， as shown in the Fig. 11 for stick number 15. Without the

Conﬁgurations of Eyeriss， NullHop， NEURAghe， CNN_A1， fpgaCon-

ISB module， each IFM stick would have to be loaded from DRAM

vNet， and Aristotle 卷积神经网络 accelerators are the ones that were
used

memory up to nine times， but by using ISB memory it is only nec-

in original papers \[19，23，25，29，30，32\] ， in order to enable easy
com-

essary to load each IFM stick exactly once from external DRAM。

parison. As for the NVDLA \[51\] ， it comes with an Excel sheet esti-

This means that for the example shown in the Fig. 11 ， it is pos-

mator， which allows specifying different conﬁguration options. We

sible to reduce DRAM memory traﬃc when transferring IFM data，

have used the one presented in Table 2 since the CoNNA accel-

up to nine times. Fig. 11 also shows the concept of replacing al-

erator is mainly intended for usage in low-cost， 边-computing

ready used IFM sticks within the ISB module with the new ones。

devices with limited computing resources。

## The new IFM stick that is written in the ISB memory by the WC

module is shown in dark gray color in the Fig. 11 .

4\. 1. Generating CoNNA 卷积神经网络 结构 instances needed for

## OSM module is used to collect output data from the RCU mod-

comparison with previous work

ule， pack it into appropriate blocks in the form of output 特征

map sticks， in order to maximize the data throughput between the

The CoNNA 卷积神经网络 accelerator was modeled as a standard soft-

accelerator and the external DRAM memory， and compress these

IP core， using SystemVerilog hardware description language， with

OFM sticks on-the-ﬂy. ZRL encoder module is used to compress se-

many conﬁguration parameters， allowing easy generation of differ-

quences of zeros within the OFM sticks， which will contain a sig-

ent CoNNA instances with the same underlying 结构. In or-

niﬁcant number of zero values if the ReLU 激活 function is

der to create a complete 卷积神经网络 acceleration system based on the

being used. Compressed OFM sticks， containing only non-zero val-

CoNNA accelerator following four steps， shown in Fig. 12 ， need to

ued OFM points and their locations within original， uncompressed

be performed：

OFM data， are then written into either external DRAM memory or

internal on-chip cache memory， as shown in the Fig. 3 . For more 1.
训练 and pruning selected 卷积神经网络 network – for this task， any

details about the encoding algorithm that is used to compress OFM
ﬁne-grained pruning algorithm can be used， and also any avail-

data， please refer to Horowitz \[46\] . able deep learning framework，
like Caffe， TensorFlow， Keras， and

Matlab， can be used to train and prune selected 卷积神经网络 network。

4\. Experiments We have implemented the ﬁne-grained pruning algorithm
de-

scribed in Section 2 within Keras deep learning framework， and

The performance of the CoNNA 结构 was compared with used Keras for 训练
and pruning 卷积神经网络 networks presented in

the following seven previously proposed 卷积神经网络 hardware accelera-
Table 1 .

tors. MIT’s Eyeriss \[19\] is an ASIC 卷积神经网络 hardware accelerator
and 2. Selecting optimal ﬁxed-point number representation and gener-

it was chosen because it is de-facto a standard reference for com- ating
a description of pruned 卷积神经网络 in a format recognizable by

parison， used in many 卷积神经网络 hardware accelerator papers. NullHop
the CoNNA 卷积神经网络 accelerator – pruned and retrained ﬂoating-

\[32\] is an FPGA-based 卷积神经网络 accelerator using the “IFM zero
skip- point 卷积神经网络 model needs to be converted into a 16-bit ﬁxed-

ping” technique. NEURAghe \[23\] is a closely coupled 卷积神经网络 hard-
point model because CoNNA uses ﬁxed-point arithmetic blocks

ware accelerator with ARM processor employing a cooperative het- for
implementing all arithmetic operations required to process

erogeneous computing approach to 卷积神经网络 acceleration， intended to
卷积神经网络 network. After this conversion， the model which is still

be used with ARM-based FPGA families， like Xilinx Zynq or Zyn-
represented in a format that depends on selected deep learning

qUltrascale families. CNN_A1 \[29\] is a reconﬁgurable 卷积神经网络
acceler- framework needs to be converted into a 卷积神经网络 structural
descrip-

ator based on using 卷积神经网络-speciﬁc Instruction Set 结构， and tion
linked list， presented in Section 3. 1 ， in order to be “un-

嵌入 parallel computation and data reuse parameters in the derstandable”
by the CoNNA 卷积神经网络 accelerator. This 卷积神经网络 struc-

instructions. fpgaConvNet 卷积神经网络 accelerator \[25\] is the
accelerator tural description linked list will actually be generated in
a bi-

created using HLS techniques targeting FPGA devices. Aristotle ac- nary
format， which can be downloaded into operating memory

<div class="page-break">

</div>

R. J. R. Struharik， B. Z. Vukobratovi ´c and A. M. Erdeljan et al. /
Microprocessors and Microsystems 73 （2020） 102991 17

Table 2

Important Characteristics of reference 卷积神经网络 accelerators used
for comparison。

Accelerator/参数 Supported 卷积神经网络 Layers Number of MAC Units
Operating Frequency Arithmetic 精确率

Eyeriss \[19\] Conv only 168 200 MHz 16-bit

NullHop \[32\] Conv， Pooling， Fully-Connected 128 60 MHz 16-bit

NVDLA \[51\] All 32 100 MHz 16-bit

NEURAghe \[23\] All 864 140 MHz 16-bit

CNN_A1 \[29\] Conv， Pooling， Fully-Connected 216 150 MHz 16-bit

fpgaConvNet \[25\] Conv， Pooling， Fully-Connected 164， 198 125 MHz
16-bit

Aristotle \[30\] Conv， Pooling， Fully-Connected 198 214 MHz 8-bit

![](/workspace/CoNNa_zh_media/a83ec62a9e85fa8e96a747b6a886cb6a7f9d10df.jpg)

Fig. 12. Development ﬂow for the 卷积神经网络 acceleration using the
CoNNA 卷积神经网络 accelerator。

and processed by the CoNNA 卷积神经网络 accelerator. We have devel- Fig.
12 . Vivado Design Suite 2018. 1 has been used to perform

oped Python-based translator software that takes pruned 卷积神经网络
synthesis and implementation of the complete 卷积神经网络 acceleration

Keras model， performs ﬂoating-point to ﬁxed-point conversion， system，
with default Vivado synthesis and implementation set-

determines the optimal 16-bit number format for representing tings.
Implementation results for six different conﬁgurations of the

all 卷积神经网络 related data and creates a binary version of
卷积神经网络 struc- CoNNA accelerator that have been used in the
experiments are

tural description linked list. shown in Table 3 .

3\. Integrating conﬁgured instance of the CoNNA accelerator IP From
Table 3 it can be seen that all conﬁgurations have been

core using Vivado IP Integrator and modifying software appli-
successfully implemented using a Xilinx ZU9 MPSoC device. Please

cation using Xilinx SDK – this step generates a complete 卷积神经网络
notice that the number of used DSP blocks for all conﬁgura-

acceleration system using some of Xilinx PSoC or MPSoC fam- tions is
larger than the number of MAC units used by the refer-

ily devices. Within this system speciﬁed conﬁguration of the ence
卷积神经网络 hardware accelerators respectively， presented in Table 2 .

CoNNA 卷积神经网络 accelerator， with a user-deﬁned number of PBs， The
reason for this is the fact that CoNNA uses additional DSP

will be integrated. blocks inside the ISM module， but these additional
DSP blocks

4\. Programing FPGA， downloading 卷积神经网络 structural description
bi- are not used to perform computations deﬁned by the target
卷积神经网络。

nary ﬁle into operating memory， initializing and starting the All six
implementations of CoNNA 结构 have been tested

CoNNA 卷积神经网络 accelerator and measuring achievable 卷积神经网络
pro- and benchmarked on six selected 卷积神经网络 architectures
（AlexNet，

cessing performance. VGG16， VGG19， GoogleNet， ResNet-18， and
ResNet-50） using Xil-

inx ZCU102 development board. Based on these experiments per-

In order to perform comparison experiments， the CoNNA 卷积神经网络

formance results have been extracted and compared with seven

accelerator has been conﬁgured in six different conﬁgurations，

reference 卷积神经网络 accelerator architectures （Eyeriss， NullHop，
NVDLA，

speciﬁed in Table 3 . The idea was to use CoNNA conﬁgurations

NEURAghe， CNN_A1， fpgaConvNet， and Aristotle）.

that are as close as possible to the conﬁgurations of the reference

accelerators from Table 2 ， in terms of the number of computing

elements （MAC units）， operating frequency and number represen- 4. 2.
Performance comparison results

tation， to enable as fair as possible performance comparison。

All six instances of the CoNNA 结构 have been imple- The results of
performance comparison experiments are pre-

mented targeting Xilinx ZU9 MPSoC device， using the ﬂow from sented in
Table 4 . For each of the reference accelerators， perfor-

Table 3

FPGA Resources required to implement CoNNA instances used in
experiments。

参数/Conﬁguration CoNNA_C1 CoNNA_C2 CoNNA_C3 CoNNA_C4 CoNNA_C5 CoNNA_C6

Number of Processing Blocks 168 128 32 256 216 198

Slice LUTs 142，470 112，550 35，042 267，424 244，464 225，132

Number of BRAMs 364 284 86 596 516 480

Number of DSPs 177 137 41 277 237 219

Operating Frequency 200 MHz 60 MHz 100 MHz 140 MHz 150 MHz 214 MHz

<div class="page-break">

</div>

## 18 R. J. R. Struharik， B. Z. Vukobratovi ´c and A. M. Erdeljan et al. / Microprocessors and Microsystems 73 （2020） 102991

Table 4

Performance comparison with reference 卷积神经网络 accelerators。

结构 卷积神经网络 结构 Latency \[ms/frame\] Frame Rate \[frames/sec\]
Performance \[GOp/sec\] Effective eﬃciency \[%\]

（a） Eyeriss AlexNet 115. 3 34. 69 46. 14 68. 66%

VGG-16 4309. 5 0. 69 21. 36 31. 79%

（a） CoNNA_C1 AlexNet 14. 20 70. 40 93. 74 139. 50%

VGG-16 102. 77 9. 73 301. 17 448. 17%

NullHop VGG-16 2269. 00 0. 44 13. 62 88. 67%

VGG-19 2439. 00 0. 41 16. 10 104. 82%

CoNNA_C2 VGG-16 375. 94 2. 66 82. 33 536. 03%

VGG-19 436. 68 2. 29 89. 94 585. 56%

NVDLA AlexNet 263. 01 3. 80 5. 51 86. 09%

GoogleNet 384. 76 2. 60 8. 21 128. 28%

ResNet-50 1843. 51 0. 54 4. 17 65. 16%

CoNNA_C3 AlexNet 96. 90 10. 32 14. 97 233. 91%

GoogleNet 202. 02 4. 95 15. 63 244. 18%

ResNet-50 431. 03 2. 32 17. 91 279. 84%

NEURAghe VGG-16 236. 61 5. 52 170. 86 238. 36%

ResNet-18 - 6. 67 23. 97 33. 44%

CoNNA_C4 VGG-16 127. 71 7. 83 242. 36 338. 11%

ResNet-18 56. 05 17. 84 64. 10 89. 42%

CNN_A1 AlexNet 144. 93 6. 90 10. 01 15. 45%

VGG-16 1639. 34 0. 61 18. 88 29. 13%

CoNNA_C5 AlexNet 21. 12 47. 35 68. 71 106. 03%

VGG-16 145. 14 6. 89 213. 26 329. 10%

fpgaConvNet AlexNet 52. 40 19. 08 27. 68 65. 90%

VGG-16 633. 00 1. 58 55. 71 132. 64%

CoNNA_C1@125MHz AlexNet 26. 00 38. 46 55. 81 132. 88%

VGG-16 180. 50 5. 54 171. 48 408. 28%

Aristotle VGG-16 364. 96 2. 74 84. 81 100. 08%

CoNNA_C6 VGG-16 101. 83 9. 82 303. 96 358. 68%

a Accelerating only convolutional layers。

mance data in terms of input instance processing latency， frame
achievable frame rate and effective eﬃciency. When compared

rate， average compute power and eﬃciency are presented. The per- with
MIT’s Eyeriss 卷积神经网络 accelerator， it can be seen that the CoNNA

formance of each of the reference accelerators is followed by a 结构
achieves higher frame rates and higher eﬃciency for

performance of the appropriate CoNNA 结构 conﬁguration， both
卷积神经网络 networks used. In the case of AlexNet 卷积神经网络
acceleration，

speciﬁed in Table 3 ， which was conﬁgured in the identical or clos-
CoNNA is capable of reaching a 2. 03 times higher frame rate than

est possible conﬁguration as the reference accelerator， to allow a
Eyeriss. In the case of VGG16 卷积神经网络 acceleration， this
improvement

fair comparison. Performance data for the reference 卷积神经网络
hardware is even greater， CoNNA is able to reach a 14. 10 times higher
frame

accelerators were taken from papers that introduced these archi- rate
than Eyeriss. Also， it can be seen that CoNNA is able to achieve

tectures \[19 ， 23 ， 25 ， 29 ， 30 ， 32\] . For the NVDLA
accelerator supplied much higher effective eﬃciency， even above 100%
due to the com-

Excel sheet estimator was used to generate achievable performance
pressed data processing， than Eyeriss. The main reason for this im-

data. provement lies in the fact that the CoNNA 结构 is able to

In the case of Eyeriss 卷积神经网络 accelerator， performance data is
take advantage of compressed kernel/weight and 特征 maps pro-

related to the processing of convolutional layers only， since Ey-
cessing， due to using the “All zero-skipping” technique， and Eyeriss

eriss cannot accelerate pooling and fully-connected layers. Also， is
not。

input instance processing latency is reported only when Eyeriss When
compared to NullHop， CoNNA is also able to achieve bet-

processes input images in batches， with batch sizes of 4 and 3， ter
performance， but this time improvement is not as dramatic as

for AlexNet and VGG16 CNNs respectively. Performance data for in the
case of the Eyeriss accelerator but is more consistent. CoNNA

all other reference accelerators is related to processing complete
achieves 6. 05 and 5. 58 times faster frame rates than NullHop when

CNNs， since all other architectures， as well as CoNNA， can process
processing VGG16 and VGG19 CNNs respectively. The main reason

all layers from a 卷积神经网络. Also， reported input instance
processing la- for this is that the NullHop accelerator is also being
able to take

tencies in these cases refer to the latency of processing a single
advantage of the sparsity present in the input 特征 maps. But

input image， i. e. working with a batch size of 1. because NullHop is
not able to take advantage of the sparsity of

Average computing performance for all architectures has been
卷积神经网络 weights， because it is not designed to process compressed

calculated as the total number of operations needed to process CNNs，
its performance is lower than CoNNA’s. In the case of VGG-

one input image by the target dense 卷积神经网络 network， multiplied by
19 卷积神经网络， NullHop is also able to reach effective eﬃciency
values

the time required to process one input image. Effective eﬃciency greater
than 100%， due to the fact that it is able to skip all compu-

was calculated as the achievable average computing performance tations
where input 特征 map values are equal to zero。

divided by the theoretical peak compute performance （calculated The
CoNNA 结构 is also superior to NVDLA， for all three

as the number of available MAC units multiplied by the operat- CNNs used
in experiments. CoNNA is able to reach 2. 72， 2. 15

ing frequency）. Values closer to 100% mean that the 卷积神经网络
accelera- and 4. 91 times higher frame rates than NVLDA， when
accelerat-

tor is more eﬃciently using available compute power. Values above ing
AlexNet， GoogleNet and ResNet-50 CNNs respectively. However，

100% mean that the 卷积神经网络 accelerator is employing some optimiza-
the improvement over NVLDA is smaller than in the case of Eye-

tion technique during 卷积神经网络 processing， effectively reducing the
re- riss and NullHop， because NVLDA is using the Winograd algorithm

quired number of computations. to speed-up the convolution calculation
process， which is suﬃcient

From Table 4 it can be seen that the CoNNA 结构 out- to allow NVDLA to
go above 100% of effective eﬃciency in case of

performs all reference 卷积神经网络 accelerator architectures in terms
of processing GoogleNet 卷积神经网络 network。

<div class="page-break">

</div>

R. J. R. Struharik， B. Z. Vukobratovi ´c and A. M. Erdeljan et al. /
Microprocessors and Microsystems 73 （2020） 102991 19

Table 5

CoNNA_C3 performance breakdown for AlexNet 卷积神经网络。

层 Active PBs Total latency \[us\] Processing latency \[us\] PB eﬃciency
\[%\] Number of operations \[GOp\] Performance \[GOp/sec\]

CONV1 32 （100%） 60，689. 63 27，668. 73 45. 60% 0. 21083 3. 47

POOL1 4 （12. 5%） 2274. 32 1574. 48 8. 65% 0. 00126 0. 55

CONV2_1 32 （100%） 9049. 99 8151. 04 98. 12% 0. 22395 24. 75

CONV2_2 32 （100%） 9049. 99 8151. 04 98. 12% 0. 22395 24. 75

POOL2 4 （12. 5%） 1029. 36 596. 72 7. 24% 0. 00078 0. 76

CONV3 32 （100%） 5409. 80 4497. 29 83. 13% 0. 29904 55. 28

CONV4_1 32 （100%） 1700. 80 1342. 00 78. 90% 0. 11214 65. 93

CONV4_2 32 （100%） 1700. 80 1342. 00 78. 90% 0. 11214 65. 93

CONV5_1 32 （100%） 1207. 34 968. 14 80. 19% 0. 07476 61. 92

CONV5_2 32 （100%） 1207. 34 968. 14 80. 19% 0. 07476 61. 92

POOL3 4 （12. 5%） 138. 61 46. 45 4. 19% 0. 00016 1. 15

FC6 4 （12. 5%） 1943. 70 1902. 74 12. 24% 0. 07550 38. 84

FC7 4 （12. 5%） 886. 74 845. 78 11. 92% 0. 03355 37. 83

FC8 4 （12. 5%） 583. 58 573. 58 12. 29% 0. 00819 14. 03

When compared with the NEURAghe 卷积神经网络 accelerator， CoNNA Table 5
presents the following data for each AlexNet 卷积神经网络 层。

is also able to achieve better performance， being 1. 42 and 2. 67 The
“Active PB” ﬁeld holds the number of Processing Blocks， out

times faster in terms of achievable frame rates. The NEURAghe of 32
available， which are actually being active during the process-

结构 is able to achieve effective eﬃciency that is signif- ing of the
current 卷积神经网络 层. The “Total Latency” stands for the

icantly higher than 100% in the case of processing the VGG-16 total time
needed to process a 卷积神经网络 层， including the time re-

卷积神经网络 network. This is curious since the authors of NEURAghe ar-
quired to preload 卷积神经网络 coeﬃcients/weights， perform required
com-

chitecture didn’t mention that they are using any 优化 putations， and
store the ﬁnal results. The “Processing Latency” is

technique， like zero skipping or Winograd algorithm， during
卷积神经网络 the time that is actually spent performing all computations
deﬁned

processing. Furthermore， there is a signiﬁcant drop in the effec- in
the current 卷积神经网络 层， excluding any preparatory steps and is

tive computing eﬃciency， of around 7 times， when the ResNet-18
therefore always shorter than the “Total Latency” time. The “PB Ef-

卷积神经网络 network is accelerated using NEURAghe. This drop， although
ﬁciency” value is calculated as the ratio of the “Processing Latency”

not that signiﬁcant， is visible in CoNNA’s performance also. This and
the “Total Latency”， multiplied with the percentage of active

can probably be partially explained by the fact that the ResNet- PBs for
the current 卷积神经网络 层. The “Number of Operations” is the

## 18 卷积神经网络 network is the network with many shallow convolutional total number of arithmetic operations required to be performed in

layers， where the number of convolutional kernels is signiﬁcantly order
to process the current 卷积神经网络 层. The “Performance” is the

lower than available 864 MAC units in NEURAghe and 256 PBs in
computational throughput for the current 卷积神经网络 层， calculated as

CoNNA_C4 conﬁguration. This means that， at least when CoNNA is the
ratio of the “Number of Operations” and the “Total Latency”.

concerned， there will be a signiﬁcant number of idle PBs during From
Table 5 it can be seen that the CoNNA 结构 is be-

the processing of these shallow convolutional layers， as described ing
able to maintain high utilization of PBs over all convolutional

in Section 3. 1 . This idling PB ineﬃciency can only be partially com-
layers， irrelevant of their size and kernel characteristics. The eﬃ-

pensated by the “All zero-skipping” technique and is the main rea-
ciency of processing convolutional layers is extremely important

son why CoNNA reaches only 89. 42% effective eﬃciency when pro- since
they are the most computationally demanding 层 type in

cessing the ResNet-18 卷积神经网络 network. modern CNNs. It can also be
seen that， as we move to deeper con-

The CoNNA 结构 is clearly superior to the CNN_A1 ar- volutional layers
of the 卷积神经网络， PB eﬃciency starts to drop. This is

chitecture， reducing instance classiﬁcation times by a factor of 6. 86
because more time has to be spent in preloading kernel coeﬃcient

and 11. 30 in the case of processing AlexNet and VGG-16 CNNs re- values
into PBs before actual computation can start. The reason

spectively. A similar observation holds in the case of fpgaConvNet for
this is that the depths of convolutional layers increase as we

and Aristotle architectures. Compared with the fpgaConvNet archi- go
deeper in the 卷积神经网络， resulting in more kernel weights data that

tecture CoNNA 结构 is able to reach frame rates that are has to be
preloaded into PBs. Since CoNNA uses limited bandwidth

2\. 02 and 3. 08 times higher. Similar to the NEURAghe 结构， data bus
to transfer data to the PBs， it takes increasingly more time

in the case of fpgaConvNet 结构， there is a signiﬁcant differ- to
preload convolutional kernel data as the size of convolutional

ence in the effective eﬃciency when processing AlexNet and VGG- kernels
increases。

## 16 卷积神经网络 networks. Furthermore， the eﬃciency of the fpgaConvNet From Table 5 it can also be seen that the PB eﬃciency

结构 when processing VGG-16 卷积神经网络 is higher than 100%， in- drops
signiﬁcantly when CoNNA is processing pooling and fully-

dicating that the fpgaConvNet 结构 must be using some op- connected
layers. Processing of these 层 types is data movement

timization technique to speedup 卷积神经网络 processing， although
authors intensive， so available data bandwidth of CoNNA’s data bus is
again

of fpgaConvNet didn’t comment on this in the original paper. Fi- the
main limiting factor. However， these layers are not computa-

nally， in the case of the Aristotle 结构， this improvement tionally
intensive， as can clearly be seen from Table 5 ， so this inef-

is slightly better， CoNNA is reaching 3. 58 times faster frame rates
ﬁciency is not signiﬁcantly degrading the total performance of the

when processing the VGG-16 卷积神经网络 network， compared with Aristo-
CoNNA 结构. From Table 5 it can be seen that the number

tle’s performance. of operations for all pooling and fully-connected
layers in the case

of AlexNet 卷积神经网络 constitutes only 8% of the total number of com-

putations， but it takes CoNNA only 9. 45% of total 卷积神经网络 compute

4\. 3. CoNNA 卷积神经网络 computational performance analysis time to
compute these layers. This means that there is an increase

of only 1. 45% in total 卷积神经网络 compute time despite very low PB
ef-

ﬁciency values attained during pooling and fully-connected 层 In order
to better understand how the CoNNA 结构 pro-

processing. cesses CNNs， and where the bottlenecks are Table 5 presents
de-

tailed execution information for the CoNNA_C3 conﬁguration when
Finally， Fig. 13 presents data throughput waveforms measured

on CoNNA’s Input Stream and Output Stream interfaces， in the it is
accelerating AlexNet 卷积神经网络。

<div class="page-break">

</div>

## 20 R. J. R. Struharik， B. Z. Vukobratovi ´c and A. M. Erdeljan et al. / Microprocessors and Microsystems 73 （2020） 102991

![](/workspace/CoNNa_zh_media/d60cdeed0fd21a81be349538612c0ed9658796f9.jpg)

Fig. 13. Data throughput waveforms over CaNNA’s ISI and OSI interfaces
in case of MobileNet V1 卷积神经网络 processing using CoNNA_C3
Instance： a） During the processing of ﬁrst

## 10 MobileNet V1 卷积神经网络 Layers； b） During processing 5th Depthwise Convolutional 层； c） During the processing of 6th Pointwise Convolutional 层。

case when MobileNet V1 卷积神经网络 is being accelerated. The CoNNA Fig.
13 b presents data throughput waveforms during the pro-

conﬁguration that was used in this experiment was CoNNA_C3 cessing of
the 5th depthwise convolutional 层 from MobileNet

from Table 3 . Please notice that each 图 actually presents V1
卷积神经网络. Analyzing the ISI interface throughput waveform it can

data throughputs on both ISI and OSI interfaces. Redline shows be
observed that it is composed out of eight almost identical

data throughput over the OSI interface， which is actually the data
segments. The reason why there are exactly eight almost iden-

throughput of writing output 特征 map data to the operating tical
segments is related to the characteristic of the 5th depth-

memory. Blue and green lines together represent data throughput wise
convolutional 层 and used conﬁguration of the CoNNA ac-

over the ISI interface. Blueline shows the data throughput during
celerator in the following way. The number of different convolu-

the loading of convolutional kernel coeﬃcients， which precedes ev-
tional kernels used within the 5th depthwise convolutional 层

= ery convolutional kernel group calculation phase. Finally， the green
from MobileNet V1 卷积神经网络 equals Kernel_Num 256. On the other

line shows data throughput during the IFM data processing phase， hand，
the CoNNA_C3 conﬁguration uses 32 Processing Blocks， i. e。

= during which IFM data is loaded into ISB memory from the Input Num_PB
32. This means that the loop L3 from Algorithm 3 dis-

Stream Manager module。

In the Fig. 13 a data throughput on CoNNA’s ISI and OSI inter- Algorithm
3

faces during the processing of the ﬁrst 10 layers from MobileNet
Pseudo-code of convolutional 层 “All zero-skipping” processing
algorithm， imple-

mented in the CoNNA 结构. V1 卷积神经网络 is shown. The reason why data
throughput during com-

plete MobileNet V1 卷积神经网络 processing was not shown is that pro- =
\< ++ L1： for （ x 0； x OFM_Width； x ）

= \< ++ cessing times of latter layers from MobileNet V1 are so short，
com- L2： for （ y 0； y OFM_Height； y ）

= \< += L3： for （kn 0； kn Kernel_Num； kn Num_PB） pared to
processing times of initial layers， that nothing could be

W1： while （PBs_Busy（））

seen on the 图. From Fig. 13 a processing periods of different

∗ ∗ += + {OFM\[ x \]\[y\]\[kn\] Calc_Next_NZPT（CIFM\[ x S h： x S h K
ernel \_W idth -

MobileNet V1 layers are clearly visible. Processing starts with the ∗
∗ + 1\]\[y Sv： y Sv Kernel_Height-1\]\[： \]，

× processing of the ﬁrst convolutional 层， which uses 3 3 ker- CKM\[ kn
\]\[： \]\[： \]\[： \]）；

\+ += OFM\[ x \]\[y\]\[kn 1\] nels to transform the input image，
designated by the yellow seg-

∗ ∗ + Calc_Next_NZPT（CIFM\[ x S h： x S h K ernel \_W idth -

ment. After this 层 is completed ﬁrst depthwise convolutional

∗ ∗ + 1\]\[y Sv： y Sv Kernel_Height-1\]\[： \]，

层， designated by the purple segment， is processed next. Next

\+ CKM\[ kn 1\]\[： \]\[： \]\[： \]）； …

comes the ﬁrst pointwise convolutional 层， designated by the or- + +=
OFM\[ x \]\[y\]\[kn Num_PB-1\]

∗ ∗ + ange segment in the Fig. 13 a， which is actually a standard
convo- Calc_Next_NZPT（CIFM\[ x S h： x S h K ernel \_W idth -

∗ ∗ + 1\]\[y Sv： y Sv Kernel_Height-1\]\[： \]， ×

lutional 层 that uses 1 1 kernels to process the input 特征

\+ CKM\[ kn N um \_PB-1\]\[： \]\[： \]\[： \]）； }

map. After this 层， a succession of depthwise and pointwise con-

volutional 层 pairs follows， which is clearly visible by alternat-

ing purple and orange segments in the Fig. 13 a. Because dataﬂows

cussed in Section 3. 1 will， in this case， be unrolled with the par-

× required for processing convolutional 层 with 3 3 kernels，

tial unrolling factor of 32. This means that 32 different depthwise

× depthwise convolutional 层 with 3 3 kernels and pointwise

convolutions， from a total of 256， will be calculated in parallel， by

× convolutional 层 with 1 1 kernels differ， data throughputs

available PBs. In order to calculate all 256 different depthwise con-

over ISI and OSI interfaces during the processing of these 层

volutions， the same process has to be repeated 8 times， each time

types differ also， which can clearly be visible on Fig. 13 a。

using a different group of 32 depthwise convolutions. At the begin-

From Fig. 13 a it can be seen that data throughput over the

ning of the processing of each group of convolutions， correspond-

OSI interface is relatively constant during 卷积神经网络 processing，
without

ing convolutional kernels need to be loaded into Local Memory

sudden bursts. In contrast， data throughput over the ISI interface

modules of each PB. During this short period of time， there is a

has a more irregular waveform， with different patterns and bursts

sudden rise in the data throughput over the ISI interface， shown

of activity， depending on the 层 type that is being processed by

by the blue line in the Fig. 13 b. After kernel preloading is com-

the CoNNA accelerator. It can be seen that there are periodic peaks

pleted， the processing of IFM data can commence. At the begging

of high throughput values， which actually correspond with periods

IFM data is fetched at a high rate， indicated by sudden peaks in

of convolutional kernel loading into Local Memory modules within

the green line， immediately following the peaks on the blue line

PBs and initial ﬁlling of the ISB memory with IFM data. During

in the Fig. 13 b， because ISB cache memory inside the Input Stream

the actual convolution calculation process throughput over the ISI

Interface module is initially empty and is being ﬁlled up by the

interface is signiﬁcantly lower， because of high IFM data reuse by

Write Controller module. Once IFM data reuse becomes active， IFM

adjacent convolutions， as explained in Section 3. 2. 2 .

data throughput falls sharply， because the WC module now has to

<div class="page-break">

</div>

R. J. R. Struharik， B. Z. Vukobratovi ´c and A. M. Erdeljan et al. /
Microprocessors and Microsystems 73 （2020） 102991 21

wait for RCU to consume individual IFM sticks from the ISB cache using
the next IFM bundle. In theory， the expected number of non-

before replacing them， which can also be visible by inspecting the zero
product terms required to compute one convolution is directly

green line from the Fig. 13 b. Since the 5th depthwise convolutional
dependant on the percentage of non-zero values within the current

× 层 uses 3 3 convolutional kernels， a theoretical decrease in the
kernel and IFM bundle， as the following equation shows

data throughput should be around nine times compared with the

= · · Num \_ MACs \_ Ideal K ernel \_ W idth K ernel \_ Height K ernel
\_ Depth

throughput during the initial ﬁlling of the ISB cache memory. How-

· · P \_ IF MNZ P \_ KMNZ （5）

ever， since CoNNA is processing compressed data and performs

zero-skipping， actual data throughput is higher and more irregu- where
P_IFMNZ is the percentage of non-zero valued points in the

lar， since compression ratios and distributions of zeros vary from
input 特征 map bundle， and P_KMNZ is the percentage of non-

one IFM stick to another. zero valued convolutional coeﬃcients in the
convolutional ker-

Finally， Fig. 13 c presents data throughput waveforms during the nel.
In this ideal scenario， each convolution computation operation

processing of a different 卷积神经网络 层 type， this time it is the 6th
would require identical time to complete

pointwise convolutional 层 from MobileNet V1 卷积神经网络. In this
case，

= · v Con olution \_ Compute \_ T ime \_ I deal N um \_ MACs \_ Ideal T
（6）

there are 16 almost identical segments， because this 层 contains clk

## 512 different convolutional kernels. What can also be seen from where T is the period of the clock signal used to synchronize

clk

Fig. 13 c is that the required throughput over the ISI interface dur-
CoNNA’s operation。

ing the actual convolution calculation process is now signiﬁcantly
However， due to the irregularity of non-zero values distribu-

higher compared with one from Fig. 13 b. The reason for this is that
tion within each kernel and IFM bundle， the actual number of

× in this case， CoNNA is actually computing 1 1 convolutions， so
non-zero product terms can and will probably vary for each ker-

there is no IFM stick reuse. nel/IFM bundle pair， leading to a
different convolution compute

time for each active PB. Since CoNNA must wait until all active

## PBs complete their convolution computation process before pro-

4\. 4. Analysis of the impact of non-zero distribution of the sparse

ceeding with the next group of convolutions， the actual compute

weights and activations on the CoNNA 结构 卷积神经网络 processing

time for each group of Num_PB convolutions will be longer than

eﬃciency

the ideal convolution computation time， as the following equation

shows

## Since CoNNA operates directly on sparse kernel and input fea-

v ture maps during 卷积神经网络 processing， and since non-zero values
dis- Con olution \_ Compute \_ T ime \_ Actual

tribution within the kernel and IF maps， in general， don’t follow { =
·

max Num \_ MACs \_ Actua l T

i clk

any regular pattern， it would be interesting to investigate how this ∈
i Active \_ PBs

≥ affects CoNNA’s 卷积神经网络 processing eﬃciency. When the CoNNA ar- v
Con olution \_ Compute \_ T ime \_ Ideal （7）

chitecture is concerned， irregular distribution of non-zero values

## In order to investigate how severe this increase in compute

within the kernel and IF maps can potentially affect CoNNA’s
卷积神经网络

time could be， a series of experiments have been performed， us-

processing eﬃciency in two following ways：

ing convolutional kernels of different sizes， with varying percent-

ages of non-zero values present within the kernel and IFM bun- 1.
Irregular non-zero values distribution within the kernel and IF

dle. The results of these experiments are shown in the Fig. 14 . maps
can lead to a different number of non-zero product terms

Each of the graphs， shown in the Fig. 14 ， shows how the convolu- that
have to be computed when the same IFM bundle is being

tion computation time changes， depending on the actual percent-
convolved with different convolutional kernels. Since CoNNA

age of non-zero values present in the kernel and IFM bundle， for has to
wait until all Processing Blocks complete their convolu-

× × × × × × × × × × 3 3 32， 3 3 64， 3 3 128， 3 3 256， 3 3 512，

tion computation process before proceeding with the computa-

× × and 3 3 1024 convolution kernels. In all these experiments

tion of the next group of convolutions， some PBs will be idling，

percentage of non-zero values that were present in the kernel and
waiting for the slowest one to ﬁnish its work， and the actual

IFM bundle has been varied independently， within the \[10% - 100%\]
convolutional 层 processing time would be prolonged. This

range. issue was discussed in Section 3. 1 when Algorithm 3 for
卷积神经网络

What can be observed in Fig. 14 is that there is indeed an in-
processing， which is actually being implemented by the CoNNA

crease in the actual convolution computation time， compared to 结构，
was presented。

the ideal theoretical estimate. This computational time increase is 2.
Irregular non-zero values distribution within the kernel and IF

more severe， as the percentage of non-zero values present in the maps
can lead to a situation where there could actually be no

kernel and IFM bundle decreases. However， this increase in the non-zero
product terms in the current non-zero product term

computational time is gradual for high and moderate percentages search
window， as the Data Fetcher module searches for the

of the non-zero kernel and IFM bundle values and rises sharply next
non-zero product term to forward to the Computing Unit。

only when the percentage of the non-zero kernel and IFM bun- This
situation will result in one additional idle clock cycle for

dle values approaches relatively small values， less than 30%. Since the
Computing Unit during product term computation and will

the achievable percentages of non-zero values after 卷积神经网络 pruning
prolong the time required to compute one convolution opera-

are rarely lower than 30% \[ 19 ， 43 \]， this means that the increase
in tion。

the convolution computation time will be acceptable， being less

Let us ﬁrst analyze the severity of the non-equal convolution than 10%，
for most of current 卷积神经网络 architectures. Furthermore， from

computation times of different PBs on the CoNNA’s 卷积神经网络
processing Fig. 14 it can also be observed that the maximum increase in
the

eﬃciency. Since CoNNA computes Num_PB different convolutions convolution
computation time decreases， as larger convolutional

in parallel， Num_PB being the number of Processing Blocks that kernels
are being used. For example， maximum increase in the

× × are available within the current CoNNA instance， if different PBs
convolution computation time for 3 3 32 kernel can be as

have to compute a different number of non-zero product terms， high as
54%， but as we move to larger kernels it quickly decreases

× × × × this can lead to different convolution computation times for
differ- to 36%， 22%， 17%， 11%， and 8%， in case of 3 3 64， 3 3
128，

× × × × × × ent PBs. In this scenario， some of PBs could be idling，
waiting for 3 3 256， 3 3 512， and 3 3 1024 convolutional ker-

the other PBs to complete their convolution computation process， nels
respectively. Most of state-of-the-art 卷积神经网络 architectures domi-

before proceeding with the next convolution computation process nantly
use deep convolutional layers， having only several relatively

<div class="page-break">

</div>

## 22 R. J. R. Struharik， B. Z. Vukobratovi ´c and A. M. Erdeljan et al. / Microprocessors and Microsystems 73 （2020） 102991

![](/workspace/CoNNa_zh_media/8e3dadb3bbee4de5c8be7658aeecf816170323b3.jpg)

× × × × Fig. 14. Relative increase in convolution 层 processing time due
to irregular distribution of non-zero kernel and IFM bundle values for：
a） 3 3 32 kernel； a） 3 3 32

× × × × × × × × × × kernel； b） 3 3 64 kernel； c） 3 3 128 kernel；
d） 3 3 256 kernel； e） 3 3 512 kernel； f） 3 3 1024 kernel。

shallow layers at the beginning of the 卷积神经网络 network. This is a
very uct term search effective， the Data Fetcher module performs a par-

favorable situation for the CoNNA 结构， meaning that the allel search，
checking a number of candidate product terms con-

relative increase in processing time will usually be well below 10%
currently， in what is called a “Search Window”. This is necessary

from ideal， theoretical processing time， for most current 卷积神经网络
ar- in order to keep the Computing Unit busy since if the non-zero

chitectures. At the end of this section， a detailed analysis of the
product term detection process would have been implemented se-

actual increase in the 卷积神经网络 层 processing time due to irregu-
quentially， the result would be an extremely ineﬃcient convolution

lar distribution of non-zero kernel and IFM bundle values for two
calculation process， where the Computing Unit would be idle most

standard 卷积神经网络 networks， AlexNet and VGG-16， will be presented
in of the time， effectively canceling any processing speedup due to

order to further justify this conclusion. zero skipping. The question is
how to select the optimal size of

Next， let us analyze the impact of the irregular distribution of this
“Search Window”， in order to minimize the number of CU idle

non-zero kernel and IFM bundle values on the eﬃciency of non- clock
cycles， while keeping required logic resources for DF module

zero product term detection logic， located within the Data Fetcher
implementation as low as possible. Intuitively， as this “Search Win-

module， as shown in Figs. 8 and 9 . As explained in Section 3. 2. 1 ，
dow” becomes larger， the probability of not detecting at least one

the Data Fetcher module searches for the non-zero product terms non-zero
product term within it would decrease. In order to in-

and forwards them to the Computing Unit which performs the ac- vestigate
how the probability of having “Empty Search Windows”

tual convolution computation process. To make this non-zero prod-
depends on the “Search Window” size and the probability of non-

<div class="page-break">

</div>

R. J. R. Struharik， B. Z. Vukobratovi ´c and A. M. Erdeljan et al. /
Microprocessors and Microsystems 73 （2020） 102991 23

![](/workspace/CoNNa_zh_media/59973c7c2183f4bf0ad5262c29a24f6a7d34faa6.jpg)

Fig. 15. Average percentage of “Empty Search Windows” as the function of
percentage of non-zero values present in kernel and IFM bundle in the
case of data fetcher

module with no built-in FIFO， for four different sizes of “Search
Window” 参数。

![](/workspace/CoNNa_zh_media/82ae7a1c52aee0e7f90c5a6beeb0dfdf62da179f.jpg)

Fig. 16. Average percentage of “Empty Search Windows” as the function of
percentage of non-zero values present in kernel and IFM bundle in the
case of data fetcher

module with built-in ideal FIFO， for four different sizes of “Search
Window” 参数。

zero kernel and IFM bundle values， a number of experiments have cant
percentage of “Empty Search Windows”， at least 20%， during

been performed， and the results are shown in Figs. 15–17 . the
computation of individual convolutions. As we increase the size

Fig. 15 shows the average percentage of “Empty Search Win- of “Search
Window”， the percentage of “Empty Search Windows”

dows” as a function of the percentage of kernel and IFM bundle starts to
fall and in the case of the “Search Window” size of 32， it

non-zero values， for four sizes of the “Search Window”， 4， 8， 16 and
stays below 10% even when there are only 50% of non-zero valued

## 32 product terms wide. From Fig. 15 it can be seen that the size of elements in the kernel and IFM bundle. On the other hand， when

“Search Window” plays a signiﬁcant role in the average percentage there
is a small percentage of non-zero values in kernel and IFM

of “Empty Search Windows”. For the “Search Window” size of 4， bundle，
less than 20%， the percentage of “Empty Search Windows”

even when kernel and IFM bundle contain a very large number of becomes
very high for all considered “Search Window” sizes， going

non-zero valued elements， more than 90%， there will be a signiﬁ- even
above 90% in the case of “Search Window” size of 4。

<div class="page-break">

</div>

## 24 R. J. R. Struharik， B. Z. Vukobratovi ´c and A. M. Erdeljan et al. / Microprocessors and Microsystems 73 （2020） 102991

![](/workspace/CoNNa_zh_media/5c35fe73824b269876d27a45c698bef490c3614d.jpg)

Fig. 17. Average percentage of “Empty Search Windows” as the function of
percentage of non-zero values present in kernel and IFM bundle in the
case of data fetcher

module with “Search Window” size of 16， with built-in FIFO of depth 0，
4， 8， 16 and 32。

In the experiments， resulting in the average percentage of the
following equation

“Empty Search Windows” surfaces， shown in the Fig. 15 ， Data

1

· = P \_ IF MNZ P \_ KMNZ （9） Fetcher module was operating without a
built-in FIFO， which re-

4

sults in the simplest possible design. However， because of the ir-

which is clearly visible in the Fig. 16 . However， the beneﬁt of us-

regular distribution of non-zero kernel and IFM bundle values， the

ing a FIFO inside Data Fetcher is clearly visible， since it can sig-
actual number of detected non-zero product terms in each “Search

niﬁcantly reduce the average percentage of “Empty Search Win-

Window” will probably vary， so there will be situations wherein

dows” for all “Search Window” sizes， up to a theoretical limit， as

the current “Search Window” there is more than one non-zero

explained above. product term detected， but in the subsequent “Search
Windows”

What can also be observed from Fig. 16 is as we increase the there are
no non-zero product terms. In this scenario， a simple

“Search Window” size， the intersection curve between the “Empty

## Data Fetcher module without a FIFO will not be able to avoid

Search Windows” surface and the zero “Empty Window Percent- idle clock
cycles， since it cannot start searching for non-zero prod-

age” plane moves steadily to the right corner. If we could use the uct
terms in the next “Search Window” until all non-zero prod-

“Search Window” of inﬁnite size we could completely eliminate

uct terms， detected in the current “Search Window” are computed。

all “Empty Search Windows”， but of course this is not possible in

Using Data Fetcher with an internal FIFO， this situation could be

practice. avoided. It would be interesting to analyze how the size of
this

In the experiments presented in the Fig. 16 ， an inﬁnitely deep

FIFO impacts the average percentage of “Empty Search Windows”，

FIFO was assumed， which cannot be used in practice. Therefore， it

and can a large enough FIFO completely eliminate “Empty Search

would be interesting to analyze how a more realistic FIFO of ﬁ-
Windows”. In order to investigate this， a new set of experiments

nite depth， reduces the percentage of “Empty Search Windows”.

has been conducted， this time with the Data Fetcher module that

Fig. 17 presents the average percentage of “Empty Search Win-

uses an inﬁnitely large FIFO， and the results of these experiments

dows” in the case when the Data Fetcher module with the “Search are
shown in the Fig. 16 .

Window” size of 16 is used， for ﬁve different depths of built-in As can
be seen from Fig. 16 ， even using an inﬁnitely deep FIFO

FIFO， 0， 4， 8， 16 and 32。

cannot remove all “Empty Search Windows”. This was to be ex-

From Fig. 17 is can be seen as the depth of used FIFO increases， pected
since when the average expected number of non-zero prod-

the average percentage of the “Empty Search Windows” surface ap- uct
terms falls below the value of one per current “Search Win-

proaches the ideal average percentage of the “Empty Search Win-

dow” size， even ideal FIFO cannot eliminate all “Empty Search Win-

dows” surface， shown in the Fig. 16 . However， from Fig. 17 can be
dows”. This is clearly visible in the Fig. 16 . When the following in-

clearly seen that using even a FIFO of depth 32 is enough to reach
equality is satisﬁed

the ideal average percentage of the “Empty Search Windows” sur-

1

face. Because of this， in the current version of the CoNNA architec- \<
“ Search W indow Size ” （8）

· P \_ IF MNZ P \_ KMNZ

ture， Data Fetcher modules were conﬁgured to use “Search Win-

non-zero product terms will be， on average， separated by more dow” of
size 16， with a built-in FIFO of depth 32。

than “Search Window Size” zero product terms and even ideal FIFO What
Figs. 14 and 17 show is that when there is a high per-

will not be able to eliminate all “Empty Search Windows”. For ex-
centage of zeros present in the kernel and/or input 特征 map， a

ample， in the case of “Search Window” size of 4， the average per-
signiﬁcant degradation of CoNNA’s performance could be present，

centage of “Empty Search Windows” surface intersects with the either
because of non-balanced number of non-zero product terms

zero “Empty Window Percentage” plane at the circle deﬁned by that are
present in the convolutional operations that are being

<div class="page-break">

</div>

R. J. R. Struharik， B. Z. Vukobratovi ´c and A. M. Erdeljan et al. /
Microprocessors and Microsystems 73 （2020） 102991 25

![](/workspace/CoNNa_zh_media/7712db05375da1318850d811e816e508dc51647a.jpg)

Fig. 18. Increase in 层 compute times for AlexNet 卷积神经网络， due to
various effects of irregular distribution of non-zero values in kernel
and input 特征 maps： a） Percentage

increase in individual 层 compute time； b） Absolute 层 processing
time。

computed in parallel by available PBs， or because of “Empty Search 2.
层 compute time when the non-balanced number of non-

Windows” that are present during the non-zero product term zero product
terms that are present in the convolutional op-

search process. However， when working with realistic 卷积神经网络
archi- erations that are being computed in parallel by available PBs is

tectures， the situation where there is a high percentage of zeros taken
into account，

present in the kernel and/or input 特征 map usually happens 3. 层
compute time when the increase in compute time due to

only within the layers that are located deep inside the 卷积神经网络，
and “Empty Search Windows” is taken into account， when the Data

mostly within the fully-connected layers. Please notice that com-
Fetcher module uses a 16 product terms wide “Search Window”

pute times of these layers contribute only slightly to the total com-
but is not using a built-in FIFO， and

pute time of complete 卷积神经网络. Therefore， any increase in the
compute 4. 层 compute time when the increase in compute time due to

times of deep convolutional or fully-connected layers due to irreg-
“Empty Search Windows” is taken into account， when the Data

ular non-zero values distribution within kernel and input 特征 Fetcher
module uses a 16 product terms wide “Search Window”

maps of these layers should result in only a small increase in the and a
32-deep built-in FIFO is being used。

total 卷积神经网络 compute time。

Figs. 18 and 19 present the results of these experiments for

In order to analyze the magnitude of the increase of total 卷积神经网络

AlexNet and VGG-16 卷积神经网络 networks respectively. In both Figures，

compute time， due to irregular non-zero values distribution within

two graphs are presented. First， showing the percentage increase

kernel and input 特征 maps， in the case of realistic 卷积神经网络 net-

in the individual 层 compute times， and the second， showing the

works， AlexNet and VGG-16 卷积神经网络 networks have been used. For

absolute 层 compute times。

both of these CNNs， individual 层 compute times have been

From Fig. 18 and 19 it can be seen that the 层 processing

computed， for several scenarios：

time increase due to a non-balanced number of non-zero product

1\. ideal 层 compute time when the increase in the 层 com- terms that
are present in the convolutional operations is not signif-

pute time due to the irregular non-zero values distribution is icant.
For two selected 卷积神经网络 networks， when convolutional layers

not taken into account， are considered， the maximum processing time
increase is less than

<div class="page-break">

</div>

## 26 R. J. R. Struharik， B. Z. Vukobratovi ´c and A. M. Erdeljan et al. / Microprocessors and Microsystems 73 （2020） 102991

![](/workspace/CoNNa_zh_media/38f2c80ebeede9a8125de25789872b0b5eafb3f0.jpg)

Fig. 19. Increase in 层 compute times for AlexNet 卷积神经网络， due to
various effects of irregular distribution of non-zero values in kernel
and input 特征 maps： a） Percentage

increase in individual 层 compute time； b） Absolute 层 processing
time。

5% compared to ideal compute time. When fully-connected layers
eliminated， or reduced， for almost all convolutional layers， and re-

are concerned， this increase is slightly higher and can reach as high
duced for all fully-connected layers。

as 13%， as is the case with the FC8 VGG-16 层. However， as can Based
on all experiments that were performed in order to an-

be observed in Figs. 18 b and 19 b， total compute times of fully- alyze
the effects of irregular non-zero values distribution within

connected layers for both AlexNet and VGG-16 CNNs are almost kernel and
input 特征 maps on the CoNNA 结构 卷积神经网络

negligible in comparison with convolutional 层 compute times，
processing time， it can be concluded that the irregular non-zero

so this increase in fully-connected 层 processing times doesn’t values
distribution has indeed an effect on the increase of CoNNA’s

signiﬁcantly change the total 卷积神经网络 processing time. In fact，
the to- total 卷积神经网络 processing time， but that this increase is
relatively small，

tal 卷积神经网络 processing time increase due to a non-balanced number
of not exceeding 3% of ideal 卷积神经网络 processing time。

non-zero product terms in the case of AlexNet and VGG-16 CNNs

is only 0. 95% and 2. 57% respectively. 5. Conclusion

When the individual 层 processing times increase due to the

“Empty Search Windows” is considered， from Figs. 18 and 19 it In this
paper， a novel 卷积神经网络 hardware accelerator， CoNNA， has

can be observed that it is more severe， and can reach the val- been
proposed. CoNNA is a coarse-grained reconﬁgurable hardware

ues as high as 80%， for FC6 and FC7 VGG-16 layers. Also， it can 结构
capable of accelerating complete pruned and com-

be seen that this increase is also signiﬁcant for the convolutional
pressed CNNs， employing the “All zero-skipping” technique to de-

layers which are located deeper within the 卷积神经网络 network， and
can crease required 卷积神经网络 processing time by skipping all
ineffectual

reach the values of 40%. However， once more， the contribution of
operations during convolutional， pooling and fully-connected 层

all these layers to the total 卷积神经网络 processing time， especially
the processing. It can be used to accelerate convolutional， depthwise

contribution of fully-connected layers， is small in comparison with
convolutional， pooling， fully-connected， concatenation and adding

the contribution of bigger convolutional layers located closer to the
layers of the target 卷积神经网络 network. CoNNA is designed to process

beginning of the 卷积神经网络 network. This is visible when the increase
in compressed CNNs， as well as input 特征 maps， which seems to

the total 卷积神经网络 processing time， due to “Empty Search Windows”
for enable achieving higher processing performance values when com-

AlexNet and VGG-16 CNNs is computed. In the case of AlexNet to- pared to
some of the previously proposed 卷积神经网络 accelerator solu-

tal processing time increase is 2. 94%， and for the VGG-16 network，
tions. CoNNA is designed to act as a co-processor soft-IP core that

it is even smaller， only 1. 10%. is connected to a host processor
within contemporary SoC or PSoC

From Figs. 18 and 19 ， the beneﬁt of using a built-in FIFO within
architectures. The CoNNA 结构 has been implemented us-

Data Fetcher modules can also be seen. When FIFO is being used， + ing
Xilinx ZynqUtrascale FPGA family and compared with seven

the resulting increase in 层 processing times can be completely
previously proposed hardware 卷积神经网络 accelerators. The CoNNA archi-

<div class="page-break">

</div>

R. J. R. Struharik， B. Z. Vukobratovi ´c and A. M. Erdeljan et al. /
Microprocessors and Microsystems 73 （2020） 102991 27

tecture， when conﬁgured to use identical number of MAC units \[21\] J.
Qiu ， J. Wang ， S. Yao ， K. Guo ， B. Li ， E. Zhou ， J. Yu ， T.
Tang ， N. Xu ， S. Song ，

Y. Wang ， Going deeper with embedded fpga platform for convolutional
neural and operating at the same clock frequency as previously proposed

network， in： Proceedings of the 2016 ACM/SIGDA International Symposium
on

MIT’s Eyeriss， NullHop and NVIDIA’s NVDLA， NEURAghe， CNN_A1，

Field-Programmable Gate Arrays， Monterey， 2016， pp. 26–35 .

fpgaConvNet， and Deephi’s Aristotle 卷积神经网络 accelerators， enables
up \[22\] Z. Liu ， Y. Dou ， J. Jiang ， J. Xu ， S. Li ， Y. Zhou ，
Y. Xu ， Throughput-Optimized FPGA

accelerator for deep convolutional neural networks， ACM Trans.
Reconﬁgurable to 14. 10， 6. 05， 4. 91， 2. 67， 11. 30， 3. 08 and 3.
58 times faster 卷积神经网络 ex-

Technol. Syst. 10 （3） （2017） 17 .

ecution of standard 卷积神经网络 networks respectively。

\[23\] P. Meloni ， A. Capotondi ， G. Deriu ， M. Brian ， F. Conti ，
D. Rossi ， L. Raffo ， L. Benini ，

NEURAghe： exploiting CPU-FPGA synergies for eﬃcient and ﬂexible
卷积神经网络 infer-

ence acceleration on Zynq SoCs， ACM Trans. Reconﬁgurable Technol. Syst.
11

Declaration of Competing Interest （3） （2018） 18 .

\[24\] N. Shah ， P. Chaudhari ， K. Varghese ， Runtime programmable
and memory band-

width optimized FPGA-Based coprocessor for deep convolutional neural
net- The authors declare that they have no known competing ﬁnan-

work， IEEE Trans. Neural Netw. Learn. Syst. 29 （12） （2018）
5922–5934 .

cial interests or personal relationships that could have appeared to

\[25\] S. I. Venieris ， C. S. Bouganis ， fpgaConvNet： mapping regular
and irregular convo-

inﬂuence the work reported in this paper. lutional neural networks on
FPGAs， IEEE Trans. Neural Netw. Learn. Syst. （2018）

（2018） 1–17 Early Access .

\[26\] J. Cheng ， J. Wu ， C. Leng ， Y. Wang ， Q. Hu ， Quantized
卷积神经网络： a uniﬁed approach

References to accelerate and compress convolutional networks， IEEE
Trans Neural Netw

Learn Syst 29 （10） （2017） 4730–4743 .

\[27\] M. Motamedi ， P. Gysel ， S. Ghiasi ， PLACID： a platform for
FPGA-based accelera-

\[1\] Y. LeCun ， Y. Bengio ， G. Hinton ， Deep learning， Nature 521
（7553） （2015）

tor creation for DCNNs， ACM Tran. Multimed. Comput. Commun. Appl. 13
（4）

436–4 4 4 .

（2017） 62 Article No .

\[2\] K. Fukushima ， Neocognitron： a self-organizing neural network
model for a

\[28\] Y. Choi ， D. Bae ， J. Sim ， S. Choi ， M. Kim ， L. S. Kim ，
Energy-Eﬃcient design of

mechanism of pattern recognition unaffected by shift in position， Biol.
Cybern。

processing element for 卷积神经网络， IEEE Trans. Circuit. Syst。

36 （4） （1980） 193–202 .

II 64 （11） （2017） 1332–1336 .

\[3\] A. Krizhevsky ， I. Sutskever ， G. E. Hinton ， ImageNet
classiﬁcation with deep con-

\[29\] X. Chen ， Z. Yu ， A ﬂexible and energy-eﬃcient 卷积神经网络
volutional neural networks， in： Advances in Neural Information
Processing Sys-

acceleration with dedicated ISA and accelerator， IEEE Trans. Very Large
Scale tems， Lake Tahoe， 2012， pp. 1097–1105 .

Integr. （VLSI） Syst. 26 （7） （2018） 1408–1412 . \[4\] R. Girshick
， R. -. C. N. N. Fast ， in： Proceedings of the IEEE Conference on
Computer

\[30\] K. Guo ， S. Han ， S. Yao ， Y. Wang ， Y. Xie ， H. Yang ，
Software-Hardware code- Vision and Pattern Recognition - CVPR， 15，
2015， pp. 1440–1448 .

sign for eﬃcient neural network acceleration， IEEE Micro. 37 （2）
（2017） 18– \[5\] Jonathan Long ， Evan Shelhamer ， Trevor Darrell ，
Fully convolutional networks

25 . for semantic segmentation， in： Proceedings of the IEEE Conference
on Com-

\[31\] J. Albericio ， P. Judd ， T. Hetherington ， T. Aamodt ， N. E.
Jerger ， A. Moshovos ， puter Vision and Pattern Recognition - CVPR，
15， 2015， pp. 3431–3440 .

Cnvlutin： ineffectual-neuron-free deep neural network computing， in：
2016 \[6\] L. Deng ， J. Li ， J. -. T. Huang ， K. Yao ， D. Yu ， F.
Seide ， M. Seltzer ， G. Zweig ， X. He ，

ACM/IEEE 43rd Annual International Symposium on Computer 结构 J.
Williams ， Y. Gong ， Recent advances in deep learning for speech
research at

（ISCA）， Seoul， 2016， pp. 1–13 . microsoft， in： IEEE International
Conference on Acoustics， Speech and Signal

\[32\] A. Aimar ， H. Mostafa ， E. Calabrese ， A. Rios-Navarro ， R.
Tapiador-Morales ， Processing （ICASSP）， Vancouver， 2013， pp.
8604–8608 .

I. A. Lungu ， M. B. Milde ， F. Corradi ， A. Linares-Barranco ， S. C.
Liu ， T. Delbruck ， \[7\] C. Chen ， A. Seff， A. Kornhauser ， J.
Xiao ， Deepdriving： learning affordance for

NullHop： a ﬂexible 卷积神经网络 accelerator based on sparse

direct perception in autonomous driving， in： Proceedings of the IEEE
Interna-

representations of 特征 maps， IEEE Trans. Neural Netw. Learn. Syst.
（2018）

tional Conference on Computer Vision， Santiago， 2015， pp. 2722–2730 .

1–13 Early Access .

\[8\] A. Esteva ， B. Kuprel ， R. A. Novoa ， J. Ko ， S. M. Swetter ，
H. M. Blau ， S. Thrun ， Derma-

\[33\] Y. Lu ， C. Wang ， L. Gong ， X. Zhou ， SparseNN： a
performance-eﬃcient accelerator

tologist-level classiﬁcation of skin cancer with deep neural networks，
Nature

for large-scale sparse neural networks， Int. J. Parallel Program. 46
（4） （2018）

542 （7639） （2017） 115–118 .

648–659 .

\[9\] D. Silver ， A. Huang ， C. J. Maddison ， A. Guez ， L. Sifre ，
G. van den Driess-

\[34\] S. Zhang ， Z. Du ， L. Zhang ， H. Lan ， S. Liu ， L. Li ， Q.
Guo ， T. Chen ， Y. Chen ，

che ， J. Schrittwieser ， I. Antonoglou ， V. Panneershelvam ， M.
Lanctot ， S. Diele-

Cambricon-x： an accelerator for sparse neural networks， in： The 49th
Annual

man ， D. Grewe ， J. Nham ， N. Kalchbrenner ， I. Sutskever ， T.
Lillicrap ， M. Leach ，

IEEE/ACM International Symposium on Microarchitecture， Taipei， 2016，
p. 20。

K. Kavukcuoglu ， T. Graepel ， D. Hassabis ， Mastering the game of Go
with deep

Article No .

neural networks and tree search， Nature 529 （7587） （2016） 4 84–4 89
.

\[35\] S. Han ， X. Liu ， H. Mao ， J. Pu ， A. Pedram ， M. A.
Horowitz ， W. J. Dally ， EIE： eﬃ- \[10\] K. Simonyan， A. Zisserman，
Very deep convolutional networks for large-scale

cient 推理 engine on compressed deep neural network， in： Proceedings
image recognition， arXiv preprint， arXiv： 1409. 1556 ， 2014。

of the 43rd International Symposium on Computer 结构， Seoul， 2016，
\[11\] NVIDIA Volta 结构 Whitepaper WP-08608-001_v1. 1， NVIDIA，
（2018），

pp. 243–254 . \[Online\]， Available： http： //images. nvidia.
com/content/volta-结构/pdf/

\[36\] A. Parashar ， M. Rhu ， A. Mukkara ， A. Puglielli ， R.
Venkatesan ， B. Khailany ， volta-结构-whitepaper. pdf .

W. J. Dally ， Scnn： an accelerator for compressed-sparse convolutional
neural \[12\] NVIDIA Jetson TX2 Delivers Twice the Intelligence to the
边， NVIDIA， （2017），

networks， in： 2017 ACM/IEEE 44th Annual International Symposium on
Com- \[Online\]， Available： https： //devblogs. nvidia. com/jetson-
tx2- delivers- twice-

puter 结构 （ISCA）， Toronto， 2017， pp. 27–40 . intelligence-边/ .

\[37\] S. Anwar， W. Sung， Compact Deep Convolutional Neural Networks
With Coarse \[13\] Intel® Stratix® 10 Variable 精确率 DSP Blocks User
Guide， Intel FPGA

Pruning， ArXiv preprint， arXiv： 1610. 09639 ， （2016）. Group，
（2017）， \[Online\]， Available： https： //www. intel.
com/content/dam/www/

\[38\] H. Li， A. Kadav， I. Durdanovic， H. Samet， H. P. Graf，
Pruning ﬁlters for eﬃcient
programmable/us/en/pdfs/literature/hb/stratix- 10/ug- s10- dsp. pdf .

convnets， arXiv preprint， arXiv： 1608. 08710 ， （2016）. \[14\] E.
Nurvitadhi ， S. Subhaschandra ， G. Boudoukh ， G. Venkatesh ， J. Sim
， D. Marr ，

\[39\] J. H. Luo ， J. Wu ， W. Lin ， Thinet： a ﬁlter level pruning
method for deep neural

R. Huang ， J. OngGeeHock ， Y. T. Liew ， K. Srivatsan ， D. Moss ，
Can FPGAs beat

network compression， in： The IEEE International Conference on Computer
Vi-

GPUs in accelerating next-generation deep neural networks？ in：
Proceedings

sion （ICCV’17）， Venice， 2017， pp. 5058–5066 .

of the ACM/SIGDA International Symposium on Field-Programmable Gate Ar-

\[40\] Y. He ， X. Zhang ， J. Sun ， Channel pruning for accelerating
very deep neural

rays - FPGA， 17， Monterey， 2017， pp. 5–14 .

networks. ， in： International Conference on Computer Vision
（ICCV’17）， Venice，

\[15\] Y. Chen ， T. Luo ， S. Liu ， S. Zhang ， L. He ， J. Wang ， L.
Li ， T. Chen ， Z. Xu ， N. Sun ，

2017， pp. 1389–1397 .

O. Temam ， DaDianNao： a machine-learning supercomputer， in：
Proceedings

\[41\] S. Anwar ， K. Hwang ， W. Sung ， Structured pruning of deep
convolutional neural

of the 47th Annual IEEE/ACM International Symposium on
Microarchitecture，

networks， ACM J. Emerg. Technol. Comput. Syst. 13 （3） （2017） 32
Article No .

Cambridge， 2014， pp. 609–622 .

\[42\] S. Han ， J. Pool ， J. Tran ， W. Dally ， Learning both weights
and connections for ef-

\[16\] K. Guo， S. Lingzhi， J. Qiu， S. Yao， S. Han， Y. Wang， and H.
Yang， Angel-eye： a

ﬁcient neural network， in： Advances in neural information processing
systems

complete design ﬂow for mapping 卷积神经网络 onto customized hardware，
in： 2016

（NIPS 2015）， Montreal， 2015， pp. 1135–1143 .

IEEE Computer Society Annual Symposium on VLSI （ISVLSI） ， Pittsburgh
（2016），

\[43\] H. Song， H. Mao， W. J. Dally， Deep compression： Compressing
deep Neural 24–29。

Networks With pruning， Trained Quantization and Huffman Coding， arXiv
\[17\] X. Wei ， C. H. Yu ， P. Zhang ， Y. Chen ， Y. Wang ， H. Hu ，
Y. Liang ， J. Cong ， Auto-

preprint， arXiv： 1510. 00149 ， （2015）. mated systolic array 结构
synthesis for high throughput 卷积神经网络 推理

\[44\] Y. Guo ， A. Yao ， Y. Chen ， Dynamic network surgery for
eﬃcient dnns， on FPGAs， in： Proceedings of the 54th Annual Design
Automation Conference，

in： Advances In Neural Information Processing Systems， Barcelona，
2016， Austin， 2017， p. 29 .

pp. 1379–1387 . \[18\] C. Wang ， L. Gong ， Q. Yu ， X. Li ， Y. Xie ，
X. Zhou ， DLAU： a scalable deep learning

\[45\] A. Erdeljan ， B. Vukobratovi ´c ， R. Struharik ， IP core for
eﬃcient zero-run accelerator unit on FPGA， IEEE Trans. Comput. -Aid.
Des. Integr. Circuits Syst. 36

length compression of 卷积神经网络 特征 maps， in： 25th
Telecommunications Forum （3） （2017） 513–517 .

（TELFOR 2017）， Belgrade， 2017， pp. 44–49 . \[19\] Y. H. Chen ， T.
Krishna ， J. S. Emer ， V. Sze ， Eyeriss： an energy-eﬃcient reconﬁg-

\[46\] M. Horowitz ， Computing’s energy problem （and what we can do
about it）， in： urable accelerator for deep convolutional neural
networks， IEEE J. Solid-State

IEEE International Solid-State Circuits Conference Digest of Technical
Papers Circuits 52 （1） （2017） 127–138 .

（ISSCC’14）， San Francisco， 2014， pp. 10–14 .

\[20\] N. Suda ， V. Chandra ， G. Dasika ， A. Mohanty ， Y. Ma ， S.
Vrudhula ， J. S. Seo ， Y. Cao ，

\[47\] A. Zhu ， T. Wang ， H. Snoussi ， Hierarchical graphical-based
human pose estima-

## Throughput-optimized OpenCL-based FPGA accelerator for large-scale convolu-

tion via local multi-resolution 卷积神经网络， AIP Adv. 8 （3）

tional neural networks， in： Proceedings of the 2016 ACM/SIGDA
International

（2018） 1–14 .

Symposium on Field-Programmable Gate Arrays， Monterey， 2016， pp.
16–25 .

<div class="page-break">

</div>

## 28 R. J. R. Struharik， B. Z. Vukobratovi ´c and A. M. Erdeljan et al. / Microprocessors and Microsystems 73 （2020） 102991

\[48\] M. Al Rahhal ， Y. Bazi ， T. Abdullah ， M. Mekhalﬁ， H.
AlHichri ， M. Zuair ， Learning a \[51\] C. Szegedy ， W. Liu ， Y. Jia
， P. Sermanet ， S. Reed ， D. Anguelov ， D. Erhan ， V. Van-

multi-branch neural network from multiple sources for knowledge
adaptation houcke ， A. Rabinovich ， Going deeper with convolutions，
in： Proceedings of the

in remote sensing imagery， Remote Sens. 10 （12） （2018） 1–18 . IEEE
Conference on Computer Vision and Pattern Recognition， Boston， 2015，

\[49\] T. Chen ， S. Lu ， J. Fan ， SS-HCNN： semi-Supervised
hierarchical convolutional pp. 1–9 .

neural network for image classiﬁcation， in： IEEE Transactions on Image
Pro- \[52\] K. He ， X. Zhang ， S. Ren ， J. Sun ， Deep residual
learning for image recognition， in：

cessing， Early Access， 2018， pp. 1–10 . Proceedings of the IEEE
Conference on Computer Vision and Pattern Recogni-

\[50\] NVIDIA Deep Learning Accelerator， \[Online\]， Available：
http： //nvdla. org/ ， tion， Las Vegas， 2016， pp. 770–778 .

（2019）.
