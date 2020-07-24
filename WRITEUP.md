# Project Write-Up

You can use this document as a template for providing your project write-up. However, if you
have a different format you prefer, feel free to use it as long as you answer all required
questions.

## Explaining Custom Layers

The process behind converting custom layers involves...

that the Model Optimizer is searching for each layer of the input model in the list of known layers before building the model's internal representation, optimizing the model, and producing the Intermediate Representation (IR).

Some of the potential reasons for handling custom layers are...

that these layers are not included into a list of known layers. If the topology contains any layers that are not in the list of known layers, the Model Optimizer classifies them as custom and reports an error.

I am going to explain 3 ways of dealing with custom layers in TensorFlow since I used a frozen pre-trained model of this framework:

There are 3 options for TensorFlow models with custom layers:

- Register those layers as extensions to the Model Optimizer. In this case, the Model Optimizer generates a valid and optimized Intermediate Representation.

- If there are sub-graphs that should not be expressed with the analogous sub-graph in the Intermediate Representation, but another sub-graph should appear in the model, the Model Optimizer provides such an option. This feature is helpful for many TensorFlow models and it is called Sub-graph Replacement 

In these cases, the sub-graph (or a single node) of initial graph is replaced with a new sub-graph (single node). The sub-graph replacement consists of the following steps:

    1. Identify an existing sub-graph for replacement
    2. Generate a new sub-graph
    3. Connect a new sub-graph to the graph (create input/output edges to the new sub-graph)
    4. Create output edges out of a new sub-graph to the graph
    5. Do something with the original sub-graph (for example, remove it)

- Experimental feature of registering definite sub-graphs of the model as those that should be offloaded to TensorFlow during inference. In this case, the Model Optimizer produces an Intermediate Representation that:

    a. Can be inferred only on CPU
    b. Reflects each sub-graph as a single custom layer in the Intermediate Representation

## Comparing Model Performance

My method(s) to compare models before and after conversion to Intermediate Representations were...

Before conversion I took the literature reference of the pre-trained model which states a speed of 31 ms vs 67.594 ms from the IR.

The accuracy is measured in mAP (Mean Average Precision) = 22, I did not know how to calculate this for the IR model since I was not uding any training or validation data. I'd be glad if you can shed some light on how to perform this over videos. 

As for the size and CPU overhead I did not found the information but I can suggest that it is similar in both models.

The difference between model accuracy pre- and post-conversion was...

36.6 ms in terms of speed which basically doubles the inference time and this might have to do with the trade-offs (Rf: https://arxiv.org/abs/1611.10012) when using the model optimizer or because the local inference is by CPU and the pre-trained model most probably was using a GPU. 

The size of the model pre- and post-conversion was...
179 Mb for the .tar.gz file
64.2 Mb for the .bin file and 
109 kb for the .xml file

The inference time of the model pre- and post-conversion was...

31 ms and 67.594 - 68,614 ms, respectivelly.

Regarding the network needs when using cloud services we might get charged depending on the time that we're sending data to the server or the fee that the provider asks for, these costs will be significantly reduced when we deploy our model at the edge.

## Assess Model Use Cases

Some of the potential use cases of the people counter app are...

Human Traffic control, keeping social distancing while the Covid-19 emergency lasts, registering the number of people coming in or out of a particular event/place.

Each of these use cases would be useful because...

From a business point of view the uses would be as follows:

- For Human Traffic control: This would help governments to detect criminals. 
- For social distancing: This would help society staying healthy for as long as it is needed under global emergency and more important would help by not saturating the hospital ICU capacity. 
- For registering the number of people at a place: For a particular business let's say a restaurant it'd be very useful to account for the busy and not busy hours e.g. whenever the facility is at its full capacity in order to optimize spaces, maybe adding or removing tables if needed.  
- For registering the number of people coming in or out: considering a retail store scenario for example it'd be useful to know how many clients they have in a given time and associate this with the monthly profits, in the case that the model recognized genders, this would give even more insights on which market target the store should focus the most. 

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a deployed edge model. The potential effects of each of these are as follows...

Lighting, model accuracy, and camera focal length/image size have different effects on a deployed edge model. The potential effects of each of these are as follows...

    - Lightning problems (incresing or decreasing) can make the model more "confused" when detecting different classes.
    - Model Accuracy would depend entirely on the end user whether they want to deploy a high or low accuracy model.
    - Camera focal length/Image Size can directly influence over the detection results e.g. if the image is out of focus or has very poor quality, then the model could hardly ever differentiate over classes, moreover if the image is pixelated/has low resolution the detection can also be compromised. On the other hand by incresing the image size/focal length the results would get probably a lot more accurate.

## Model Research

I did tried several models but finally got the TF one (ssd_mobilenet_v2_coco_2018_03_29) to work with the edge app.

Downloaded from: 
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md

http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz

I first decompressed the model using:

tar -xvf ssd_mobilenet_v2_coco_2018_03_29.tar.gz

cd to ssd_mobilenet_v2_coco_2018_03_29

Converting into IR:

Here's what I entered to convert the SSD MobileNet V2 model from TensorFlow:

python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json

Output:

[ SUCCESS ] Generated IR model.
[ SUCCESS ] XML file: /home/workspace/ssd_mobilenet_v2_coco_2018_03_29/./frozen_inference_graph.xml
[ SUCCESS ] BIN file: /home/workspace/ssd_mobilenet_v2_coco_2018_03_29/./frozen_inference_graph.bin
[ SUCCESS ] Total execution time: 76.27 seconds.


Running the ssd_mobilenet_v2_coco_2018_03_29 model: 

python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm