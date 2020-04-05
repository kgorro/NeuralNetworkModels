using System;
using System.Collections.Generic;

namespace Perceptron
{
    public class PerceptronNeuralNetwork
    {
        private List<double> weights = new List<double>();
        private List<int[]> trainingLetterPattern = new List<int[]>();
        private List<int> outputLayer = new List<int>();

        private Random randomNumbers = new Random();
        private double learningRate = 0.1;

        /*7x5 column for 0 and 1 to draw a letter + 1 bias*/
        private const int MAXINPUT = 36;
        private const int MAXALPHABET = 26;

        public  List<double> _Weights
        {
            get { return weights; }
        }

        public List<int[]> _TrainingLetterPattern
        {
            get { return trainingLetterPattern;  }
        }

        public void initializeWeights()
        {
            /*Initialize the weight: +1 with bias*/
            for(int i = 0; i < MAXINPUT; i++)
            {
                weights.Add(Math.Round(randomNumbers.NextDouble(),2));
            }
        }

        public void initializeOutputLayer()
        {
            /*
             * 26 alphabet
             * 1 as Vowel
             * 0 as Consonant
             */
            if (outputLayer.Count == MAXALPHABET)
                return;

            int[] expected = new int[]{
                1, 0, 0, 0, 1,
                0, 0, 0, 1, 0,
                0, 0, 0, 0, 1,
                0, 0, 0, 0, 0,
                1, 0, 0, 0, 0,
                0
            };

            outputLayer.AddRange(expected);
        }

        public bool addCharacterPattern(int [] pattern)
        {
            /*success*/
            trainingLetterPattern.Add(pattern);
            return true;
        }

        public bool setTrainingLayerPatter(List<int[]> letters)
        {
            trainingLetterPattern.AddRange(letters);
            return true;
        }

        public bool trainPerceptron()
        {
            try
            {
                double weightedSum = 0;
                int yValue = 0;
                int deltaValue = 0;
                int costError = 1;

                while (costError != 0)
                {
                    costError = 0;

                    for (int i = 0; i < MAXALPHABET; i++)
                    {
                        /*calculate the Total = Xi x Wi*/
                        weightedSum = 0;

                        /*traverse to each input plus the bias*/
                        for (int j = 0; j < MAXINPUT; j++)
                        {
                            if (j < (MAXINPUT - 1))
                                weightedSum += Math.Round(trainingLetterPattern[i][j] * weights[j], 5);
                            else
                                weightedSum += weights[j];
                        }

                        /*evaluate the vValue*/
                        yValue = (weightedSum <= 0) ? 0 : 1;

                        /*Calculate the delta value*/
                        deltaValue = outputLayer[i] - yValue;

                        /*calculate the error*/
                        costError += Math.Abs(deltaValue);

                        for (int j = 0; j < MAXINPUT; j++)
                        {
                            if (j < (MAXINPUT- 1))
                                weights[j] = Math.Round(weights[j] + (learningRate * deltaValue * trainingLetterPattern[i][j]), 2);
                            else
                                weights[j] = Math.Round(weights[j] + (learningRate * deltaValue), 2);
                        }

                    }
                }

            }
            catch (Exception ex)
            {
                /*training failed*/
                return false;
            }

            return true;
        }

        private double evaluateInput(int []input)
        {
            double result = 0;

            /*calculate the vvalue*/
            for (int i = 0; i < (MAXINPUT - 1); i++)
            {
                result += Math.Round(input[i] * weights[i],5);
            }

            result += weights[MAXINPUT - 1];

            return result;
        }

        public String whatTypeOfLetter(int []input)
        {
            if (input.Length < MAXINPUT - 1 || input.Length > MAXINPUT - 1)
                return "Invalid Input";

            double result = evaluateInput(input);

            if (result <= 0)
                return "Consonant";

            else
                return "Vowel";
        }
    }
}
