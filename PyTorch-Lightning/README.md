<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Medium][medium-shield]][medium-url]
[![Twitter][twitter-shield]][twitter-url]
[![Linkedin][linkedin-shield]][linkedin-url]

# PyTorch Lightning: Build your models with PyTorch and train them with PyTorch Lightning
This repository shows a couple of examples to start using PyTorch Lightning right away. PyTorch Lightning provides several functionalities that allow to organize in a flexible, clean and understandable way each component of the training phase of a PyTorch model.

If you want to know more in detail how to structure each PyTorch Lightning component, I recommend the following blog: <a href="https://towardsdatascience.com/pytorch-lightning-making-your-training-phase-cleaner-and-easier-845c4629445b?sk=d761b0e0089fb035b4526f334d6417c0"> PyTorch Lightning: Making your Training Phase Cleaner and Easier</a>

<p align="center">
<img src='img/lightning.jpg'>
</p>

<!-- TABLE OF CONTENTS -->
## Table of Contents

* [Files](#files)
* [How to use](#how-to-use)
* [Contributing](#contributing)
* [Contact](#contact)
* [License](#license)

<!-- files -->
## 1. Files
* **main_basic.py**: This file contains a "basic" implementation of PyTorch Lighting. 
* **main_tune.py**: This file contains a "more advanced" implementation of PyTorch Lightning, essentially it covers the use of some specific functionalities such as: finding automatically the optimal learning rate as well as finding the optimal batch size.


<!-- how-to-use -->
## 2. How to use
You just need to type

```SH
python main_basic.py
```

or

```SH
python main_tune.py
```

however, I recommend you to work with a virtual environment, in this case I am using pipenv. So in order to install the dependencies located in the ``Pipfile`` you just need to type:

```SH
pipenv install
```
and then

```SH
pipenv shell
```

<!-- contributing -->
## 3. Contributing
Feel free to fork the model and add your own suggestiongs.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/YourGreatFeature`)
3. Commit your Changes (`git commit -m 'Add some YourGreatFeature'`)
4. Push to the Branch (`git push origin feature/YourGreatFeature`)
5. Open a Pull Request

<!-- contact -->
## 5. Contact
If you have any question, feel free to reach me out at:
* <a href="https://twitter.com/Fernando_LpzV">Twitter</a>
* <a href="https://medium.com/@fer.neutron">Medium</a>
* <a href="https://www.linkedin.com/in/fernando-lopezvelasco/">Linkedin</a>
* Email: fer.neutron@gmail.com

<!-- license -->
## 6. License
Distributed under the MIT License. See ``LICENSE.md`` for more information.


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[medium-shield]: https://img.shields.io/badge/medium-%2312100E.svg?&style=for-the-badge&logo=medium&logoColor=white
[medium-url]: https://medium.com/@fer.neutron
[twitter-shield]: https://img.shields.io/badge/twitter-%231DA1F2.svg?&style=for-the-badge&logo=twitter&logoColor=white
[twitter-url]: https://twitter.com/Fernando_LpzV
[linkedin-shield]: https://img.shields.io/badge/linkedin-%230077B5.svg?&style=for-the-badge&logo=linkedin&logoColor=white
[linkedin-url]: https://www.linkedin.com/in/fernando-lopezvelasco/