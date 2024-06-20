{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "hGSjk7LK5MX-"
      },
      "source": [
        "# Setup (disk and dataset)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "xiN7zVwmeyI6",
        "outputId": "e2e207cd-b61f-407d-d23b-38ae0e2333a3"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Mounted at /content/drive\n"
          ]
        }
      ],
      "source": [
        "from google.colab import drive\n",
        "drive.mount('/content/drive')"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "hJ_22tEXva-A",
        "outputId": "dbe61e5a-fb7c-4962-cda6-5263ce05d714"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "cuda\n"
          ]
        }
      ],
      "source": [
        "import torch\n",
        "\n",
        "from torchvision import datasets, models, transforms\n",
        "from torch.utils.data import DataLoader\n",
        "from torch.utils.data.sampler import SubsetRandomSampler\n",
        "\n",
        "train_dir = \"/content/drive/MyDrive/ITMO/lisa-lab/dataset/train\"\n",
        "test_dir = \"/content/drive/MyDrive/ITMO/lisa-lab/dataset/test\"\n",
        "\n",
        "batch_size = 64\n",
        "\n",
        "#  NOTE: check if GPU is available\n",
        "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
        "print(device)\n",
        "\n",
        "data_transforms = {\n",
        "    \"train\": transforms.Compose(\n",
        "        [\n",
        "            transforms.Resize((224, 224)),\n",
        "            transforms.ToTensor(),\n",
        "            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),\n",
        "        ]\n",
        "    ),\n",
        "    \"test\": transforms.Compose(\n",
        "        [\n",
        "            transforms.Resize((224, 224)),\n",
        "            transforms.ToTensor(),\n",
        "            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),\n",
        "        ]\n",
        "    ),\n",
        "}\n",
        "\n",
        "\n",
        "train_dataset = datasets.ImageFolder(\n",
        "    train_dir, transform=data_transforms[\"train\"]\n",
        ")\n",
        "test_dataset = datasets.ImageFolder(test_dir, transform=data_transforms[\"test\"])\n",
        "\n",
        "\n",
        "train_loader = DataLoader(\n",
        "    train_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(range(900))\n",
        ")\n",
        "test_loader = DataLoader(\n",
        "    test_dataset, batch_size=batch_size#, sampler=SubsetRandomSampler(range(100))\n",
        ")\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "c97FkzThrDsX"
      },
      "source": [
        "# ResNet-34"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "5XHC6kP3qrlp",
        "outputId": "e5ecb182-65bb-42df-b587-1e5eba6c7902"
      },
      "outputs": [
        {
          "name": "stderr",
          "output_type": "stream",
          "text": [
            "Downloading: \"https://download.pytorch.org/models/resnet34-b627a593.pth\" to /root/.cache/torch/hub/checkpoints/resnet34-b627a593.pth\n",
            "100%|██████████| 83.3M/83.3M [00:00<00:00, 91.0MB/s]\n"
          ]
        },
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Start train 2024-06-20 17:42:51.717534\n",
            "Finish train 2024-06-20 17:42:51.719136\n"
          ]
        }
      ],
      "source": [
        "import datetime\n",
        "import torch.nn as nn\n",
        "from torchvision import datasets, models, transforms\n",
        "from sklearn.metrics import f1_score, confusion_matrix\n",
        "\n",
        "\n",
        "model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)\n",
        "num_features = model.fc.in_features\n",
        "model.fc = nn.Linear(num_features, 2)  # fake and real\n",
        "\n",
        "criterion = torch.nn.CrossEntropyLoss()\n",
        "optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)\n",
        "\n",
        "model = model.to(device)\n",
        "\n",
        "print(\"Start train\", datetime.datetime.now())\n",
        "model.train()\n",
        "print(\"Finish train\", datetime.datetime.now())\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "background_save": true,
          "base_uri": "https://localhost:8080/"
        },
        "id": "nZg4U7-owpBt",
        "outputId": "5057585b-06b6-49da-aa44-e6703f463809"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Start run epochs 2024-06-20 15:46:47.986041\n",
            "Started epoch 0 2024-06-20 15:46:47.986329\n",
            "Finished epoch 0 2024-06-20 15:53:37.740309\n",
            "Started epoch 1 2024-06-20 15:53:37.741384\n",
            "Finished epoch 1 2024-06-20 16:00:21.048438\n",
            "Started epoch 2 2024-06-20 16:00:21.048620\n",
            "Finished epoch 2 2024-06-20 16:07:07.104453\n",
            "Started epoch 3 2024-06-20 16:07:07.104643\n",
            "Finished epoch 3 2024-06-20 16:13:49.573912\n",
            "Started epoch 4 2024-06-20 16:13:49.575494\n",
            "Finished epoch 4 2024-06-20 16:20:33.693260\n",
            "Started epoch 5 2024-06-20 16:20:33.693437\n",
            "Finished epoch 5 2024-06-20 16:27:15.840804\n",
            "Started epoch 6 2024-06-20 16:27:15.842385\n",
            "Finished epoch 6 2024-06-20 16:33:56.198639\n",
            "Started epoch 7 2024-06-20 16:33:56.198853\n",
            "Finished epoch 7 2024-06-20 16:40:35.790618\n",
            "Started epoch 8 2024-06-20 16:40:35.790825\n",
            "Finished epoch 8 2024-06-20 16:47:18.990618\n",
            "Started epoch 9 2024-06-20 16:47:18.992613\n",
            "Finished epoch 9 2024-06-20 16:53:58.885110\n",
            "Started epoch 10 2024-06-20 16:53:58.885284\n",
            "Finished epoch 10 2024-06-20 17:00:43.116944\n",
            "Started epoch 11 2024-06-20 17:00:43.117146\n",
            "Finished epoch 11 2024-06-20 17:07:24.854616\n",
            "Started epoch 12 2024-06-20 17:07:24.856356\n",
            "Finished epoch 12 2024-06-20 17:14:03.778425\n",
            "Started epoch 13 2024-06-20 17:14:03.779705\n",
            "Finished epoch 13 2024-06-20 17:20:47.889576\n",
            "Started epoch 14 2024-06-20 17:20:47.889753\n",
            "Finished epoch 14 2024-06-20 17:27:25.559985\n",
            "Started epoch 15 2024-06-20 17:27:25.560304\n",
            "Finished epoch 15 2024-06-20 17:34:09.030870\n",
            "Started epoch 16 2024-06-20 17:34:09.031098\n",
            "Finished epoch 16 2024-06-20 17:40:47.557984\n",
            "Started epoch 17 2024-06-20 17:40:47.558171\n",
            "Finished epoch 17 2024-06-20 17:47:30.000862\n",
            "Started epoch 18 2024-06-20 17:47:30.002338\n",
            "Finished epoch 18 2024-06-20 17:54:06.230925\n",
            "Started epoch 19 2024-06-20 17:54:06.231107\n",
            "Finished epoch 19 2024-06-20 18:00:49.249767\n",
            "Stop run epochs 2024-06-20 18:00:49.251539\n"
          ]
        }
      ],
      "source": [
        "import datetime\n",
        "\n",
        "\n",
        "num_epochs = 20\n",
        "print(\"Start run epochs\", datetime.datetime.now())\n",
        "for epoch in range(num_epochs):\n",
        "    print(f\"Started epoch {epoch}\", datetime.datetime.now())\n",
        "    for inputs, labels in train_loader:\n",
        "        inputs = inputs.to(device)\n",
        "        labels = labels.to(device)\n",
        "        optimizer.zero_grad()\n",
        "        outputs = model(inputs)\n",
        "        loss = criterion(outputs, labels)\n",
        "        loss.backward()\n",
        "        optimizer.step()\n",
        "        model.eval()\n",
        "    print(f\"Finished epoch {epoch}\", datetime.datetime.now())\n",
        "print(\"Stop run epochs\", datetime.datetime.now())\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "background_save": true
        },
        "id": "Mf6fI1_h4W_m",
        "outputId": "60730276-519e-46f1-b94c-8558610ea4d8"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Start predictions 2024-06-20 18:00:49.463877\n",
            "Stop predictions 2024-06-20 18:15:12.369885\n"
          ]
        }
      ],
      "source": [
        "all_predictions = []\n",
        "all_labels = []\n",
        "\n",
        "print(\"Start predictions\", datetime.datetime.now())\n",
        "with torch.no_grad():\n",
        "    for inputs, labels in test_loader:\n",
        "        inputs = inputs.to(device)\n",
        "        labels = labels.to(device)\n",
        "        outputs = model(inputs)\n",
        "        _, predicted = torch.max(outputs, 1)\n",
        "        all_predictions.extend(predicted.cpu().numpy())\n",
        "        all_labels.extend(labels.cpu().numpy())\n",
        "print(\"Stop predictions\", datetime.datetime.now())"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "background_save": true
        },
        "id": "58QhQAMDwrMQ",
        "outputId": "83660915-0964-4400-efd2-c47e0d8ec1c7"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "0.3333333333333333\n",
            "[[1000    0]\n",
            " [1000    0]]\n"
          ]
        }
      ],
      "source": [
        "f1 = f1_score(all_labels, all_predictions, average=\"weighted\")\n",
        "\n",
        "conf_matrix = confusion_matrix(all_labels, all_predictions)\n",
        "\n",
        "print(f1)\n",
        "print(conf_matrix)"
      ]
    }
  ],
  "metadata": {
    "accelerator": "GPU",
    "colab": {
      "gpuType": "T4",
      "provenance": [],
      "collapsed_sections": [
        "hGSjk7LK5MX-",
        "c97FkzThrDsX"
      ]
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}