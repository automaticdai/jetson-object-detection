# jetson-object-detection

Following the success of my other project, [rpi-object-detection](https://github.com/automaticdai/rpi-object-detection), I am making this new project focusing on Nvidia Jetson. As many configurations are different on Jetson, I think this is a better way to have a new repo, rather than a hybrid integration to support multiple platforms. Due to the on-board GPU, with Jetson, I am going to produce more interesting CV-based projects.

Star the project and stay tuned!

## Hardware
This repo is developed with Jetson AGX Orin, and it should support the following hardware:

- Jetson AGX Thor (2025)
- Jetson Orin Nano Super (2024)
- Jetson AGX Orin (2023)
- Jetson Orin NX (2023)
- Jetson Orin Nano (2023)

However, there is no support for older boards, including:

- Jetson Xavier NX (2020)
- Jetson Nano (2019)
- Jetson AGX Xavier (2018)
- Jetson TX2 (2017)
- Jetson TX1 (2015)
- Jetson TK1 (2014)

Note that this does not mean the code won't be able to run on these boards --- I just don't have the time and energy to test them.

## Requirements
- Jetson Linux
- Jetpack
- Opencv2
- Pytorch

## License
MIT License
