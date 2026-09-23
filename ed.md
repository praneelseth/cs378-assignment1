For part 3, the assignment says to include "A table of your chosen Br, Bc and the resulting shared-memory usage per block." To clarify, do we calculate the shared-memory usage based on the Br, Bc values or do we figure out how to measure it?

Comment

1 Answer
Anish Biswas
Last week

2

For the Part 3 deliverable we expect you to evaluate your formulation for a couple of values of Br,Bc. Part 4(a) is where you will write down the derivation and answer follow up questions in  depth.

Comment

Angela Zhang
1w
We tried using nvcc -Xptxas=-v -lineinfo -O3 -arch=sm_80 src/flash_attention_kernel.cu -o flash_attention , but it wasn't showing the shared memory usage. Would you be able to point us in the right direction for how to measure it for part 3?


2
Reply

Anish Biswas
6d
Hi Angela, as discussed during office hours, I'm writing this down here for anyone else who might have the same question. For this deliverable, we expect an analytical sweep across different Br and Bc configurations, and the shared memory usage should be calculated based on that sweep.



1
Hello folks,

Here are some clarifications for parts of Assignment 1 that may not be clear:

3.2) Pick 3–4 different values of Br and Bc to create a table showing shared memory usage (You may assume Br = Bc). You can either measure the values experimentally or estimate them analytically. Please explain the method you used and how you arrived at the reported values.

4.b) You may assume batch size = 1 and number of heads = 32 and 128. Run experiments using these values and the suggested context lengths for each regime, along with a few different Br and Bc values, to analyze and answer the questions. Please mention which experiments you ran and explain how they led to your conclusions.


Please reach out if there are any other questions you have! 