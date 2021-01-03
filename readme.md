# Scrabble cheater

This project is a word puzzle solver for _Bananagram_ game. The project has nothing doing with original _Scrabble_ game, even though name "Scrabble cheater" may suggest so. However, Bananagram is very similar to Scrabble. The objective is to arrange randomly given letter tiles to Finnish words so that all tiles are used.

![Example solution](example_image.jpg)


## Background
My sister gave Bananagram for our family as a Christmas present. She is good at it. Because I'm better in cheating, I wrote this solver.


## Algorithm

This program uses Hastings algorithm underneath. Which is as follows:
1. Do a random move, but keep old configuration in memory. Here one letter tile moved randomly.
2. Calculate a fitness. Here fitness is evaluated by counting number of full and partial words on the table.
3. Compare how did fitness change on this move. If the new arrangement is better than previous, then keep it. If it is worse, then accept it with some probability. The worse is new arrangement, the less likely it is to be accepted.
4. If current state is not a solution, then repeat from step 1.

This algorithm does random moves and they get closer and closer to solution. Bad moves are accepted so that algorithm does not get stuck on local minima. This solver finds a solution with about 80% probability.



### Licence
The code is under MIT and the list of Finnish words is under LGPL. The word list is generated from [source](https://kaino.kotus.fi/sanat/nykysuomi/).
