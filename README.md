# Flappy-Bird_DDQN
I have used DDQN algorithm to train the Flappy Bird. After training for 24 hours, it's average score was 84(average is taken over last ten steps). It was able to achieve a max score of 384. When I stopped it's training at that time it's average score was increasing.
I have used some tweaks to make the algorithm learn faster. I have kept the background black. Used Biassed greedy policy to gain reward etc.
I have used a low spec laptop for its training. That's why it took 5 hours to beat the human average and 12 hours to have an average score above 45. If you have access to a high-end machine. I strongly encourage you to run this algorithm because you can get a sense of hyperparameter through this. This problem shows immediate effect of change of hyperparameter relative to other RL problems.

Saved model file contains parameters after the training of 5 hours. 

#### Fun video of result after random hours of training 
![henaha](https://user-images.githubusercontent.com/28859302/43942167-895385a0-9c94-11e8-94a0-e9532e548b2f.gif)
