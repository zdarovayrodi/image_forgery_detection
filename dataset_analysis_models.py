{
  "cells": [
    {
      "cell_type": "markdown",
      "source": [
        "# Google Net"
      ],
      "metadata": {
        "id": "GY2fx4p0mUmZ"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import torch\n",
        "from torchvision import datasets, transforms\n",
        "from torch.utils.data import DataLoader\n",
        "\n",
        "train_dir = \"/content/drive/MyDrive/ITMO/lisa-lab/dataset/train\"\n",
        "test_dir = \"/content/drive/MyDrive/ITMO/lisa-lab/dataset/test\"\n",
        "\n",
        "batch_size = 16\n",
        "\n",
        "# Check if GPU is available\n",
        "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
        "print(device)\n",
        "\n",
        "data_transforms = transforms.Compose([\n",
        "    transforms.Resize((224, 224)),\n",
        "    transforms.RandomHorizontalFlip(),\n",
        "    transforms.ToTensor(),\n",
        "     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])\n",
        "])\n",
        "\n",
        "train_dataset = datasets.ImageFolder(train_dir, transform=data_transforms)\n",
        "test_dataset = datasets.ImageFolder(test_dir, transform=data_transforms)\n",
        "\n",
        "train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)\n",
        "test_loader = DataLoader(test_dataset, batch_size=batch_size)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "a8NUwGdymYFD",
        "outputId": "e4d00ecb-e495-46b8-adb3-20cbe1596bce"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "cpu\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import torch\n",
        "import torch.nn as nn\n",
        "import torch.optim as optim\n",
        "import torchvision.models as models\n",
        "\n",
        "model = models.googlenet(pretrained=True)\n",
        "num_classes = len(train_dataset.classes)\n",
        "model.fc = nn.Linear(model.fc.in_features, num_classes)\n",
        "\n",
        "criterion = nn.CrossEntropyLoss()\n",
        "optimizer = optim.Adam(model.parameters(), lr=0.001)"
      ],
      "metadata": {
        "id": "r9B3Wedyp4hZ"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "num_epochs = 10\n",
        "\n",
        "model.to(device)\n",
        "\n",
        "for epoch in range(num_epochs):\n",
        "    model.train()\n",
        "    running_loss = 0.0\n",
        "\n",
        "    for inputs, labels in train_loader:\n",
        "        inputs, labels = inputs.to(device), labels.to(device)\n",
        "\n",
        "        optimizer.zero_grad()\n",
        "\n",
        "        outputs = model(inputs)\n",
        "        loss = criterion(outputs, labels)\n",
        "        loss.backward()\n",
        "        optimizer.step()\n",
        "\n",
        "        running_loss += loss.item()\n",
        "\n",
        "    print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}')\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "zsl0qqERrLNb",
        "outputId": "8183e6fa-2cb4-4ae8-cb58-1117f2716263"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1, Loss: 0.5267827959060669\n",
            "Epoch 2, Loss: 0.40176488757133483\n",
            "Epoch 3, Loss: 0.30068591141700746\n",
            "Epoch 4, Loss: 0.2989252612888813\n",
            "Epoch 5, Loss: 0.20594286198914052\n",
            "Epoch 6, Loss: 0.19569240839779378\n",
            "Epoch 7, Loss: 0.17925950498133897\n",
            "Epoch 8, Loss: 0.1636695774048567\n",
            "Epoch 9, Loss: 0.13314844775944948\n",
            "Epoch 10, Loss: 0.14673510287329555\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.metrics import confusion_matrix, f1_score\n",
        "import numpy as np\n",
        "\n",
        "model.eval()\n",
        "\n",
        "true_labels = []\n",
        "pred_labels = []\n",
        "\n",
        "with torch.no_grad():\n",
        "    for inputs, labels in test_loader:\n",
        "        inputs, labels = inputs.to(device), labels.to(device)\n",
        "        outputs = model(inputs)\n",
        "        _, predicted = torch.max(outputs, 1)\n",
        "        true_labels.extend(labels.cpu().numpy())\n",
        "        pred_labels.extend(predicted.cpu().numpy())\n",
        "\n",
        "conf_matrix = confusion_matrix(true_labels, pred_labels)\n",
        "f1 = f1_score(true_labels, pred_labels, average='weighted')\n",
        "\n",
        "print(\"Матрица ошибок:\")\n",
        "print(conf_matrix)\n",
        "print(\"F1:\", f1)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "YqAIfqUTp8yU",
        "outputId": "95e5c83a-5e1d-40f5-957d-7fdeedb3aab0"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Матрица ошибок:\n",
            "[[942  58]\n",
            " [256 744]]\n",
            "F1: 0.8414460123672111\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import seaborn as sns\n",
        "import matplotlib.pyplot as plt\n",
        "\n",
        "sns.heatmap(conf_matrix,\n",
        "            annot=True,\n",
        "            fmt='g',\n",
        "            xticklabels=['Real', 'Fake'],\n",
        "            yticklabels=['Real', 'Fake'])\n",
        "plt.xlabel(\"Prediction\", fontsize=13)\n",
        "plt.ylabel(\"Actual\", fontsize=13)\n",
        "plt.title(\"Confusion Matrix\", fontsize=13)\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 476
        },
        "id": "rnnrvOKUvu_x",
        "outputId": "12ec7ee0-5dc7-432a-ce1e-7bca1ae7c232"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 2 Axes>"
            ],
            "image/png": "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I/DDDz8gNzcXJ06cwLVr1zB69OgX3vN8deHJkyd4+PAhnjx5Ai8vL1y+fBlZWVllimHatGmlblutWjVs3rwZ9+7dQ+/evTF37ly88847mDhxYpmeSUSvxuSDqhxbW1s8fvy41O2TkpIAAE2aNCl27dm5GzduaJyvU6dOsbaOjo4AUGwuxesaOnQovL29sWjRIjg4OMDLywuLFy/GrVu3XnlvUlISGjZsCBMTzRFWExMTNGjQoNjnAl782crzuUaNGoWsrCzs2LED69atg6urK3x8fF7YPi0tDePGjVPP0ahevTqcnJwQGRkJAMjIyCjT8xs0aFCm9h06dMCMGTMQHx8PJycnrFu3rkz3E1HpMPmgKqdp06bIysoq8RertrxsVYlQiqWfEonkhdf+vf+IVCpFTEwM4uPjERoaCmNjY4SFhaFRo0YlVi5e14s+W2k+1795eHigffv2WLlyJbZu3Yr33nvvpf337NkT69evR2BgIH744Qfs378fMTEx6rkeKpWqTM+3tLQsU/v8/HwcOHAAAJCeno7k5OQy3U9EpcPkg6ocf39/AChxQmNJnv2lf/HixWLXLl26pNFGW54tvU1PTy927Vkl5t/atWuHWbNmISYmBomJibCyssLMmTNf+pw6derg6tWrxRKawsJCXLt2TeufqySjR4/GyZMnkZ2d/dIhlz///BPnzp3Dxx9/jIiICAwePBg+Pj7w9vZGUVFRsfYvS+DKKzQ0FL///jsiIiJga2uLoUOHcpULUQVg8kFVzvvvv4+GDRvi888/L3GpLAAkJCRg1apVAJ6uiLCyssKKFSs0hmseP36MFStWwNraGj169NBqjM+GA/49l+T777/H3bt3Nc49ePCg2P1vvPEGnJycSkxenufn54f79+8XS8S++eYb3L9/H/379y9P+GUydOhQzJ49G8uWLUP9+vVf2O5ZReTfFZYLFy6UWOF5tnLlVd+D0tq3bx+WLl2KwMBAhISEICoqCteuXeOcD6IKwKW2VOVYWlpi79698PX1hZ+fH3r27IkePXrA0dER9+/fxy+//IIDBw6oN7+yt7dHREQEgoKC0L59e/X+D9HR0UhMTMSaNWvUkz61pWHDhvD29saaNWsgCAJatmyJs2fPYufOnahXrx4KCgrUbRcsWICDBw+ib9++cHd3hyAI2LNnD65cuVJsA69/mz59OrZt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\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Inception-v3"
      ],
      "metadata": {
        "id": "hDrmVJW0y-Fe"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import torch\n",
        "from torchvision import datasets, transforms\n",
        "from torch.utils.data import DataLoader\n",
        "\n",
        "train_dir = \"/content/drive/MyDrive/ITMO/lisa-lab/dataset/train\"\n",
        "test_dir = \"/content/drive/MyDrive/ITMO/lisa-lab/dataset/test\"\n",
        "\n",
        "batch_size = 16\n",
        "\n",
        "# Check if GPU is available\n",
        "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
        "print(device)\n",
        "\n",
        "data_transforms = transforms.Compose([\n",
        "    transforms.Resize((299, 299)),\n",
        "    transforms.RandomHorizontalFlip(),\n",
        "    transforms.ToTensor(),\n",
        "    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])\n",
        "])\n",
        "\n",
        "train_dataset = datasets.ImageFolder(train_dir, transform=data_transforms)\n",
        "test_dataset = datasets.ImageFolder(test_dir, transform=data_transforms)\n",
        "\n",
        "train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)\n",
        "test_loader = DataLoader(test_dataset, batch_size=batch_size)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "QvolXQhcy7ST",
        "outputId": "315f17d3-b62d-4ea8-afdf-eabeffced79e"
      },
      "execution_count": 2,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "cuda\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import torch\n",
        "import torch.nn as nn\n",
        "import torch.optim as optim\n",
        "import torchvision.models as models\n",
        "\n",
        "model = models.inception_v3(pretrained=True)\n",
        "num_classes = len(train_dataset.classes)\n",
        "\n",
        "model.fc = nn.Linear(model.fc.in_features, num_classes)\n",
        "model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, num_classes)\n",
        "\n",
        "criterion = nn.CrossEntropyLoss()\n",
        "optimizer = optim.Adam(model.parameters(), lr=0.001)"
      ],
      "metadata": {
        "id": "b4D4TMzPzCX9"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "num_epochs = 10\n",
        "\n",
        "model.to(device)\n",
        "\n",
        "for epoch in range(num_epochs):\n",
        "    model.train()\n",
        "    running_loss = 0.0\n",
        "\n",
        "    for inputs, labels in train_loader:\n",
        "        inputs, labels = inputs.to(device), labels.to(device)\n",
        "\n",
        "        optimizer.zero_grad()\n",
        "\n",
        "        outputs, aux_outputs = model(inputs)\n",
        "        loss1 = criterion(outputs, labels)\n",
        "        loss2 = criterion(aux_outputs, labels)\n",
        "        loss = loss1 + 0.4 * loss2  # combining the losses from both outputs\n",
        "\n",
        "        loss.backward()\n",
        "        optimizer.step()\n",
        "\n",
        "        running_loss += loss.item()\n",
        "\n",
        "    print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "KG8fnxZMzI0e",
        "outputId": "e3e7071a-6e76-40d3-97b7-95eff7fec516"
      },
      "execution_count": 4,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1, Loss: 0.8624255726337433\n",
            "Epoch 2, Loss: 0.6020945165157318\n",
            "Epoch 3, Loss: 0.4730161460638046\n",
            "Epoch 4, Loss: 0.3734891216754913\n",
            "Epoch 5, Loss: 0.3169632055312395\n",
            "Epoch 6, Loss: 0.27574330557882787\n",
            "Epoch 7, Loss: 0.2838679488897324\n",
            "Epoch 8, Loss: 0.16892640866339206\n",
            "Epoch 9, Loss: 0.1325386668369174\n",
            "Epoch 10, Loss: 0.18145751452445985\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from sklearn.metrics import confusion_matrix, f1_score\n",
        "import numpy as np\n",
        "\n",
        "model.eval()\n",
        "\n",
        "true_labels = []\n",
        "pred_labels = []\n",
        "\n",
        "with torch.no_grad():\n",
        "    for inputs, labels in test_loader:\n",
        "        inputs, labels = inputs.to(device), labels.to(device)\n",
        "        outputs = model(inputs)\n",
        "        _, predicted = torch.max(outputs, 1)\n",
        "        true_labels.extend(labels.cpu().numpy())\n",
        "        pred_labels.extend(predicted.cpu().numpy())\n",
        "\n",
        "conf_matrix = confusion_matrix(true_labels, pred_labels)\n",
        "f1 = f1_score(true_labels, pred_labels, average='weighted')\n",
        "\n",
        "print(\"Матрица ошибок:\")\n",
        "print(conf_matrix)\n",
        "print(\"F1:\", f1)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "t2uEDDpDzPC6",
        "outputId": "2348ff13-779f-4609-b371-c0445ccc2cd1"
      },
      "execution_count": 5,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Матрица ошибок:\n",
            "[[829 171]\n",
            " [ 56 944]]\n",
            "F1: 0.8861234958080152\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import seaborn as sns\n",
        "import matplotlib.pyplot as plt\n",
        "\n",
        "sns.heatmap(conf_matrix,\n",
        "            annot=True,\n",
        "            fmt='g',\n",
        "            xticklabels=['Real', 'Fake'],\n",
        "            yticklabels=['Real', 'Fake'])\n",
        "plt.xlabel(\"Prediction\", fontsize=13)\n",
        "plt.ylabel(\"Actual\", fontsize=13)\n",
        "plt.title(\"Confusion Matrix\", fontsize=13)\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 476
        },
        "id": "AW9uRJcczQln",
        "outputId": "1a8d42be-3925-4ce3-d3e1-c7fb411d6cc9"
      },
      "execution_count": 6,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 2 Axes>"
            ],
            "image/png": "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UCgAAAqFAhcuXEBqaqqmTUxMDGQyGby8vIodN+d8EBERiUzQwyZjtra2aNiwodY5GxsbODo6as4PHToUEyZMgIODA2QyGT755BMoFAp4e3sDADp37gwvLy8MGDAA4eHhSE5OxrRp0xAaGvrSpKcoTD6IiIhE9qYbhJWVxYsXw8TEBIGBgVCpVPD398fKlSs1101NTbFnzx6MGjUKCoUCNjY2CAkJwZw5c0r0HIkgCBV+i3kzi6r6DoHIID29f1zfIRAZHPMqNcv8GSur9ddJP6P/Xq+TfsTGygcREZHIDLXyIRYmH0RERCKr8EMOr8Hkg4iISGR8qy0RERGRiFj5ICIiEhnnfBAREZGojD354LALERERiYqVDyIiIpFxtQsRERGJiqtdiIiIiETEygcREZHIjH3CKZMPIiIikRn7nA8OuxAREZGoWPkgIiISmdrIax9MPoiIiETGOR9EREQkKuOue3DOBxEREYmMlQ8iIiKRcdiFiIiIRMUdTomIiIhExMoHERGRyLjUloiIiERl3KkHh12IiIhIZKx8EBERiYyrXYiIiEhUxj7ng8MuREREJCpWPoiIiERm3HUPJh9ERESi45wPIiIiEhXnfBARERGJiJUPIiIikRl33YOVDyIiItGpdXSUxLfffovGjRtDJpNBJpNBoVDgt99+01zPyclBaGgoHB0dUalSJQQGBiIlJUWrj6SkJAQEBMDa2hrOzs6YPHky8vPzS/z5DTb5OH78OPr37w+FQoF79+4BANatW4cTJ07oOTIiIqLy56233sJXX32FhIQE/P777/D19UX37t1x6dIlAMD48eOxe/dubNmyBUePHsX9+/fRs2dPzf0FBQUICAhAbm4uYmNjsXbtWkRFRWHGjBkljsUgk49t27bB398fVlZWOHv2LFQqFQAgIyMD8+fP13N0REREb0bQ0X8l8f7776Nbt26oU6cOPD098eWXX6JSpUo4deoUMjIysHr1aixatAi+vr5o0aIFIiMjERsbi1OnTgEAoqOjcfnyZaxfvx5NmzZF165dMXfuXKxYsQK5ubklisUgk4958+YhIiIC33//PczNzTXnfXx88Mcff+gxMiIiojenq2EXlUqFzMxMreP5P9hfpaCgAJs2bUJ2djYUCgUSEhKQl5cHPz8/TZt69eqhevXqiIuLAwDExcWhUaNGcHFx0bTx9/dH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\n"
          },
          "metadata": {}
        }
      ]
    }
  ],
  "metadata": {
    "accelerator": "GPU",
    "colab": {
      "gpuType": "T4",
      "provenance": []
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