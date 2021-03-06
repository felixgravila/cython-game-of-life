#%%

import numpy as np
import cv2
import gol_worker
import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython import display
from time import sleep

N = 800
# inclusive
# MIN_NEIGHBORS_TO_SURVIVE = 2
# MAX_NEIGHBORS_TO_SURVIVE = 3
# MIN_NEIGHBORS_TO_BE_BORN = 3
# MAX_NEIGHBORS_TO_BE_BORN = 3
# RADIUS_FOR_NEIGHBORS = 1

INIT_ALIVE_PROBABILITY = 0.15
MIN_NEIGHBORS_TO_SURVIVE = 11
MAX_NEIGHBORS_TO_SURVIVE = 17
MIN_NEIGHBORS_TO_BE_BORN = 11
MAX_NEIGHBORS_TO_BE_BORN = 13
RADIUS_FOR_NEIGHBORS = 3

INIT_ALIVE_PROBABILITY = 0.15

ITERS = 100000

assert INIT_ALIVE_PROBABILITY > 0 and INIT_ALIVE_PROBABILITY <= 1

x = np.random.choice(
    2, size=(N, N), p=[1 - INIT_ALIVE_PROBABILITY, INIT_ALIVE_PROBABILITY]
).astype(np.int8)

cv2.namedWindow("game of life")
img = None
last = {"survived": 0, "died": 0, "born": 0}
for i in range(ITERS):
    print()
    img = np.array(x).astype(np.float32)
    img = (img - 1) * -1  # invert
    img = np.stack([img, img, img], axis=-1)  # convert to rgb
    img = cv2.resize(
        img, (1600, 1600), interpolation=cv2.INTER_NEAREST
    )  # resize to make sure we can see it
    text = f"{i+1:03d}/{ITERS}, {last['survived']}s {last['died']}d {last['born']}b"
    img = cv2.putText(
        img,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 190),
        1,
        cv2.LINE_AA,
    )
    cv2.imshow("game of life", img)
    if cv2.waitKey(20) & 0xFF == ord("q"):
        break

    prev_x = x
    last = gol_worker.iterate(
        x,
        min_survive=MIN_NEIGHBORS_TO_SURVIVE,
        max_survive=MAX_NEIGHBORS_TO_SURVIVE,
        min_born=MIN_NEIGHBORS_TO_BE_BORN,
        max_born=MAX_NEIGHBORS_TO_BE_BORN,
        size_n=RADIUS_FOR_NEIGHBORS,
    )
    x = last["result"]

    if (np.array(x) == np.array(prev_x)).all():
        break

cv2.destroyAllWindows()
cv2.waitKey(1)


# %%
