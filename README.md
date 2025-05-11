# CudaLifeEngine

This project is heavily based on the life engine of Max Robinson/Emergent Garden found here: https://github.com/MaxRobinsonTheGreat/LifeEngine. The goal was to take his engine, written in Javascript, and create one using CUDA to leverage parallel processing and GPU capabilities. This involved completely rethinking how many of the systems were previously implemented so that they can be elegantly and efficiently parallelized. Currently there is very limited functionality relative to the original, but "lifelike growth" can still be seen by running the code and holding down the enter key to simulate 100 iterations of the producer organism propagating across the grid.  

This project was started and completed over the course of one weekend in order to help learn CUDA, and also because I really like the original Life Engine and wanted it to run faster.
