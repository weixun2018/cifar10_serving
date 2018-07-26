#!/usr/bin/env python
# encoding=utf-8

import tensorflow as tf
from grpc.beta import implementations
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2

tf.app.flags.DEFINE_string("host", "10.30.1.3", "TensorFlow Serving server ip")
tf.app.flags.DEFINE_integer("port", 9999, "TensorFlow Serving server port")
tf.app.flags.DEFINE_string("model_name", "cifar10", "The model name")
tf.app.flags.DEFINE_integer("model_version", 1, "The model version")
tf.app.flags.DEFINE_string("signature_name", "predict_images", "The model signature name")
tf.app.flags.DEFINE_string("data_dir", "/tmp", "Inferece data directory")
tf.app.flags.DEFINE_float("request_timeout", 10.0, "Timeout of gRPC request")
FLAGS = tf.app.flags.FLAGS


def create_predict_client():
    channel = implementations.insecure_channel(FLAGS.host, FLAGS.port)
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

    return stub


def main():
    # Create gRPC client
    stub = create_predict_client()
    # Create a request for prediction
    request = predict_pb2.PredictRequest()
    request.model_spec.name = FLAGS.model_name
    if FLAGS.model_version > 0:
        request.model_spec.version.value = FLAGS.model_version
    if FLAGS.signature_name != "":
        request.model_spec.signature_name = FLAGS.signature_name

    with open('test_images/bird.jpeg', 'rb') as f:
        image = f.read()

    request.inputs['images'].CopyFrom(
        tf.contrib.util.make_tensor_proto(image, shape=[1, ]))

    # Send request
    result = stub.Predict(request, FLAGS.request_timeout)
    print(result)


if __name__ == "__main__":
    main()
