import cv2
import numpy as np
import onnxruntime as ort
import time


model = "/workspace/Face_Pipeline_V2/models/SCRFD/scrfd_10g_bnkps.onnx"

sess = ort.InferenceSession(
    model,
    providers=[
        "CPUExecutionProvider"
    ]
)

print("Providers:", sess.get_providers())


img = cv2.imread("/workspace/Face_Pipeline_V2/tests/RD_Radio1_000_frame4.jpg")

assert img is not None, "image load failed"

img = cv2.resize(img, (640,640))

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img = img.transpose(2,0,1)

img = np.expand_dims(img,0)

img = img.astype(np.float32)


input_name = sess.get_inputs()[0].name


for i in range(5):
    outputs = sess.run(
        None,
        {input_name: img}
    )


start=time.time()

for i in range(50):
    outputs = sess.run(
        None,
        {input_name: img}
    )

end=time.time()


print("avg ms:", (end-start)/50*1000)

print("output num:", len(outputs))

for o in outputs:
    print(o.shape)
