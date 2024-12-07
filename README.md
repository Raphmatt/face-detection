# pfp validator

## Setup

### Requirements

- Python 3.11
- Docker

1. Clone the project to your computer.

    ```bash
    git clone git@gitlab.com:GIBZ/students/infa3a2023/face-detection.git
    ```

2. Start the project with Docker.

    ```bash
    docker compose up --build
    ```

3. Open the Swagger page [docs](http://localhost:8000/docs).


## Interfaces

The Python FastAPI provides two interfaces:

| Request Type | Path       | Body     | Description |
|-------------|--------------------|------------------|----------------------------------------|
| **GET**     | /heartbeat         | No parameters  | Returns the status of the service    |
| **POST**    | /image/process     | A .png image    | Processes the image and sends it back|

### Parameters "image/process"

| Parameter | Default | Type | Description |
| ---- | ---- | ---- | ---- |
| bounds | False | boolean | Allow visible image edges |
| side_spacing | 0.72 | number | Distance of eyes to image edge <br> 0 = Eyes at edge <br> 0.9998 = Eyes centered from far |
| top_spacing | 0.4 | number | Vertical position of eyes <br> 0 = Eyes at top edge <br> 0.9998 = Eyes at bottom edge |
| width | 512 | integer | Width of final image |
| height | 640 | integer | Height of final image |
| binary_method | multiclass | string | Method used to remove background <br> multiclass = more accurate, slower <br> selfie = less accurate, faster |

## Process

The user sends the image from their smartphone to the Rust Service. This posts it to our FastAPI Service, which analyzes and processes the image and finally returns it. In the Rust Service, the final image is uploaded to the Google Bucket as usual.

### Architecture Diagram

![img](./files/ArchitectureDiagram.png){width=60%}

### Flow Diagram

![img](./files/FlowDiagram.png){width=30%}

## Flexibility

In our project, the individual steps of image processing were split into different functions. This makes them easy to exchange or extend.

Even in the Rust Service, the URL could simply be pointed to another service, as long as the parameters and return format don't change.

## System Components / Frameworks


![Static Badge](https://img.shields.io/badge/Mediapipe-0.10.8-lightblue?logo=google)

![Static Badge](https://img.shields.io/badge/dlib-19.24.2-green?logo=dlib)

![Static Badge](https://img.shields.io/badge/FastAPI-0.105.0-darkgreen?logo=fastapi)

## Design Decision

We decided to use Mediapipe due to its high configurability and the possibility of local execution. It also allows us to expand our knowledge of Computer Vision while deepening our Python skills.

For implementation into the existing GIBZ solution, we considered three variants:

1. Direct call via Mobile Client
2. Call in Rust Service
3. Call in Web Frontend

Finally, we chose the **second** method for the following reasons:

- no updates needed in the Mobile Client, which could potentially be more difficult to implement/deploy

- the read/write processes in the Google Bucket are kept to a minimum, as we assume that a high number of requests would lead to higher costs

## Quality of the Solution

### Advantages

- Our service is only added in the backend. The user doesn't need to perform any updates or similar and won't notice anything.
- We don't use external services like Google Vision or AWS Rekognition, but execute everything locally. This ensures student data protection.
- The performance of our solution is already adequate but could easily be scaled by adjusting the cloud environment.
- By avoiding external providers, we save costs.


### Disadvantages

- We had to invest a lot of time in learning Computer Vision and the actual implementation. This would certainly have been easier using an external service.

## Implemented

- [x] Exactly one human face should be recognizable in the photo.
- [x] The face in the photo must be completely visible.
- [x] The face must be taken frontally. The head must not be tilted too much on any axis.
- [x] The face must not be covered (e.g., by masks, pets, sunglasses, ...).


## Integration into Overall System

The integration into the overall system, as shown in the diagram, is not complex, and the required code changes were made in [our fork](https://gitlab.com/GIBZ/students/infa3a2023/profile-picture-server/-/tree/gipeFix?ref_type=heads).

## Video

https://github.com/Raphmatt/face-detection/assets/71792812/2c9d941f-e80d-4973-95be-1a36a9904cf2

## Bonus

We are particularly proud of the implementation of background removal, which uses integrated smoothing. Equally impressive is the automatic alignment of the photo depending on the angle of the face. It also takes into account how far the face is from the camera, resulting in a uniform final image.

We owe this outstanding solution to Raphael Andermatt (@raphmatt), who invested a lot of time and effort into it.
