# Bachelor's Thesis
## Federated Learning for Multi-Institutional Medical Image Segmentation.

Deep Learning has been widely used for medical image segmentation and a large number of papers have been presented recording the success of Deep Learning in this field. The performance of Deep Learning models strongly relies on the amount and diversity of data used for training. In the Medical Imaging field, acquiring large and diverse datasets is a significant challenge. Unlike photography images, labeling medical images require expert knowledge. Ideally, collaboration between institutions could address this challenge but sharing medical data to a centralized location faces various legal, privacy, technical, and data-ownership challenges. This is a significant barrier in pursuing scientific collaboration across transnational medical research institutions.

Traditionally, Artificial Intelligence techniques require centralized data collection and processing that may be infeasible in realistic healthcare scenarios due to the aforementioned challenges. In recent years, Federated Learning has emerged as a distributed collaborative AI paradigm that enables the collaborative training of Deep Learning models by coordinating with multiple clients (e.g., medical institutions) without the need of sharing raw data. Although Federated Learning was initially designed for mobile edge devices, it has attracted increasing attention in the healthcare domain because of its privacy preserving nature of the patient information.

In Federated Learning, each client trains its own model using local data, and only the model updates are sent to the central server. The server accumulates and aggregates the individual updates to yield a global model and then sends the new shared parameters to the clients for further training. In this way, the training data remains private to each client and is never shared during the learning process. Only the model’s updates are shared, thus keeping patient data private and enabling multi-institutional collaboration.

[Read More](https://github.com/avocadopelvis/BTP/blob/main/paper.pdf)

## Federated Learning Architecture
![delete](https://user-images.githubusercontent.com/92647313/178327276-dc3e960b-089a-4e95-9a3c-140d1f0a3ef8.png)

## MODELS
- U-net [Ronneberger et al. (2015)](https://arxiv.org/abs/1505.04597)
- U-net++ [Zhou et al. (2018)](https://arxiv.org/abs/1807.10165)
- Attention U-net [Oktay et al. (2018)](https://arxiv.org/abs/1804.03999)
