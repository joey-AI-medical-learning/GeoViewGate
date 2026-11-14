# GeoViewGate: Geometry-Guided Multi-View Representation Learning for Odontogenic Lesion Classification in CBCT

## Overview

This repository provides the implementation of a two-stage deep learning framework for the classification of odontogenic lesions in cone-beam computed tomography (CBCT) images. The framework aims to differentiate ameloblastoma (AME), odontogenic keratocyst (OKC), dentigerous cyst (DC), and radicular cyst (RC) through a hierarchical classification strategy. Specifically, the first stage distinguishes ameloblastoma from odontogenic cystic lesions, while the second stage further differentiates the three types of odontogenic cysts (OKC, DC, and RC).
The framework consists of two stages:

- **Stage 1: Ameloblastoma (AME) is distinguished from odontogenic cysts.

- **Stage 2: The odontogenic cysts are further classified into three subtypes: odontogenic keratocyst (OKC), dentigerous cyst (DC), and radicular cyst (RC).

## Installation

### 1. Clone the Repository

```bash
git clone <code_address>
cd GeoViewGate
```

### 2. Recommended Dependencies

```text
python==3.10
torch==2.4.1
torchvision==0.19.1
torchaudio==2.4.1
pandas==2.0.3
tensorboard==2.11.2
tqdm==4.67.1
scipy==1.10.1
einops==0.6.1
```

## Stage 1: Ameloblastoma (AME) is distinguished from odontogenic cysts.
### 1. Training

Train the first-stage classification model to distinguish ameloblastoma (AME) from odontogenic cysts (OKC, DC, and RC).:

```bash
python train.py
use model/Stage_1.py 
```

## Stage 2: The odontogenic cysts are further classified into three subtypes: odontogenic keratocyst (OKC), dentigerous cyst (DC), and radicular cyst (RC).

### 1. Training

Train the second-stage classification model to differentiate odontogenic keratocyst (OKC), dentigerous cyst (DC), and radicular cyst (RC).

```bash
python train.py
use model/Stage_2.py 
```

### 3. Testing

Evaluate the classification model on the test dataset:

```bash
python test.py
```

## Dataset Availability

The CBCT dataset used for the two-stage classification of odontogenic lesions, including ameloblastoma (AME), odontogenic keratocyst (OKC), dentigerous cyst (DC), and radicular cyst (RC), is privately held and cannot be made publicly available due to institutional data-sharing restrictions and patient privacy considerations. The source code is provided to facilitate the reproducibility of the proposed methodology.
