# Faster ML 1: Using And Understanding Onnx Runtime

Want to get your neural network to run faster at serving time? Need to deploy your Pytorch model with Tensorflow? Want to run your model natively on Android with Java? You need Onnx Runtime, friend. But what is Onnx Runtime, and how does it pull off this alchemy of faster and more flexible machine learning? 

Before diving into specifics, let's define some terms: 

* **ONNX** - [Open Neural Network eXchange](https://onnx.ai/). Open format which defines a common set of ml operators, which allow interoperability and easier hardware acceleration. Each neural net is represented as an acyclic graph with each node being a call to an operator. Is authored in [Protobuf format](https://protobuf.dev/). 
* **Intermediate Representations** - [A representation](https://web.stanford.edu/class/archive/cs/cs143/cs143.1128/handouts/230%20Intermediate%20Rep.pdf) of the source program which the compiler uses to convert into source code.
* **Execution Provder** - The extensible framework Onnx Runtime provides so that different hardware providers can optimally execute ONNX Runtime. Examples include NVIDIA's CUDA EP and TensorRT, Apple's CoreML, and Android's 'Android Neural Network API'
* **Operator Fusion** - Speeds up model execution by treating two or more operators as a single operator, thus reducing memory reads and writes. 

### Part 1: Exporting Models To Onnx

There are multiple tutorials for exporting models to Onnx format in Tensorflow, Pytorch and Jax. For sake of example, we can show a simple fine-tuned Yolo model. To export in ONNX format, simply provide 'onnx' as a parameter:

```python
yolo = YOLO("yolov8n.pt")
yolo.train(data='drive/MyDrive/teeth_yolov8_format/tooth.v1i.yolov8/data.yaml')
yolo.export(format="onnx")
```

Now, let's see how this affects inference of our Yolo model. 

### Part 2: Python - To - Python Acceleration 

TODO - YoloV8 Inference In Notebook

## Part 3: Cross - Language Deployment 

##### Build in Python, deploy in Java for Android

### Python -> Android


## Part 4: Graph Optimizations

### Operator fusion 


## Resources

#### 1. Onnx Docs 
1. https://becominghuman.ai/a-deep-dive-into-onnx-onnx-runtime-part-2-785b523e0cca
2. https://becominghuman.ai/a-deep-dive-into-onnx-onnx-runtime-part-1-874517c66ffc
3. [Onnx Runtime Github](https://github.com/microsoft/onnxruntime)
4. [Onnx Runtime Docs](https://onnxruntime.ai/docs/)
5. [Onnx Optimizer](https://github.com/onnx/optimizer)

#### 2. Intermediate Representations
1. [DLIR](https://link.springer.com/chapter/10.1007/978-3-030-05677-3_19)
2. [MILR (tensorflow)](https://www.tensorflow.org/mlir)
3. [DistIR: An Intermediate Representation and Simulator for Efficient Neural Network Distribution](https://arxiv.org/abs/2111.05426)
4. [Torch FX](https://pytorch.org/docs/stable/fx.html)
5. [Stanford CS 147 Handout](https://web.stanford.edu/class/archive/cs/cs143/cs143.1128/handouts/230%20Intermediate%20Rep.pdf)

### 3. Operator Fusion
1. [Pytorch Operator Fusion](https://towardsdatascience.com/how-pytorch-2-0-accelerates-deep-learning-with-operator-fusion-and-cpu-gpu-code-generation-35132a85bd26)
2. [Fused Softmax With Triton](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)
3. [Operator Fusion Example](https://quadric.io/2023/09/13/how-to-unlock-the-power-of-operator-fusion-to-accelerate-ai/#:~:text=Conceptually%2C%20operator%20fusion)

### 4. Profiling Changes 
1. [Netron Profiler](https://github.com/lutzroeder/Netron)

### 5. Execution Provider
1. [CUDA EP and TensorRT Execution Providers](https://developer.nvidia.com/blog/end-to-end-ai-for-nvidia-based-pcs-cuda-and-tensorrt-execution-providers-in-onnx-runtime/)
