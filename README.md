# face detection

# Todo

- [ ] Docker
  - [ ] Dockerfile
  - [ ] docker-compose.yml
- [ ] API
  - [ ] FastAPI Functionality
    - [ ] Human Face detection
      - [x] Face count detection
    - [ ] Face visibility
      - [ ] Face angle
        - [x] Detect face angle
        - [ ] Adjust face angle / threshold for face angle
      - [ ] Shoulder angle
        - [x] Detect shoulder angle
        - [x] threshold for shoulder angle
      - [ ] Face visible from the front
        - [ ] Detect face horizontal angle
        - [ ] threshold for face horizontal angle
      - [ ] Face not occluded by an object
        - [ ] Detect occlusion
    - [ ] Image optimization
      - [ ] Remove background
        - [x] Get mask of background
        - [ ] Remove background
      - [ ] Uniform cropping to the face
        - [ ] Detect face position
        - [x] Crop image to face position
        - [ ] Resize image to uniform size
        - [ ] Rotate image to uniform angle
      - [ ] Uniform aspect ratio
        - [ ] Apply uniform aspect ratio
      - [ ] Constant, absolute dimensions (e.g. 500x500px)
        - [x] Scale image to constant, absolute dimensions


# Requirements:

- Validation
  - Is a human face visible in the image?
  - Is the face completely visible?
  - Is the face visible from the front and not at an angle or from the side?
  - Is the face not occluded by an object? (e.g. Mask, Pet, Sunglasses, etc.)
- Optimization
  - Remove the background, so that the person is shown in front of a transparent background.
  - Uniform cropping of the image, so that all faces are shown in the same size, orientation and position.
  - The resulting image should be saved in a uniform aspect ratio with constant, absolute dimensions.
- Additional
  - Is the person a celebrity?

## Original Requirements


### Validierung der Profilfotos

Die Inhaltliche Validierung der Profilfotos umfasst zumindest folgende Aspekte (Mindestanforderungen):

- Auf dem Foto soll genau ein menschliches Gesicht erkennbar sein.
- Das Gesicht auf dem Foto muss vollständig sichtbar sein.
- Das Gesicht muss frontal aufgenommen sein. Der Kopf darf also in keiner Achse zu stark geneigt sein.
- Das Gesicht darf nicht durch verdeckt sein (z.B. durch Masken, Haustiere, Sonnenbrillen, ...).

### Optimierung der Profilfotos

Sofern alle relevanten Validierungen erfolgreich abgeschlossen werden konnten, sollen die Profilfotos für die weitere Verwendung optimiert werden.

#### Entfernung des Hintergrundes

Der Hintergrund soll entfernt werden. Als Endprodukt soll ein Profilfoto resultieren, in welchem die fotografierte Person freigestellt, vor einem transparenten Hintergrund abgebildet ist. Bezüglich Dateiformat existieren keine weiteren Vorgaben, ein breites Einsatzspektrum soll jedoch angestrebt werden.

#### Uniformer Zuschnitt

Alle Profilbilder sollen einen uniformen Zuschnitt erfahren, damit die fotografierten Gesichter grundsätzlich gleich gross erscheinen. Bei der Ausrichtung, Positionierung und Grösse des Gesichts sollen nach Möglichkeit geltende Standards oder Konventionen angewendet werden. Das resultierende Foto soll in einem uniformen Seitenverhältnis mit konstanten, absoluten Dimensionen persistiert werden.
