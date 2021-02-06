using System;
using System.Collections.Generic;
using System.Text;

namespace Perceptron
{
    public class PerceptronNeuralNetwork
    {
        private List<double> weights = new List<double>();
        private List<int[]> trainingLetterPattern = new List<int[]>();
        private List<int[]> trainingCharacterRecogPattern = new List<int[]>();

        private List<int> outputLayer = new List<int>();
        private List<int[]> outputCharacterRecognition = new List<int[]>();

        private Random randomNumbers = new Random();
        private double learningRate = 0.1;
        private double[,] weightMultiLayerPercep = new double[5, 36];

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

        public List<int[]> _TrainingCharacterRecognition
        {
            get { return trainingCharacterRecogPattern; }
        }

        public void initializeWeights()
        {
            /*Initialize the weight: +1 with bias*/
            for(int i = 0; i < MAXINPUT; i++)
            {
                weights.Add(Math.Round(randomNumbers.NextDouble(),2));
            }

            for (int x = 0; x < 5; x++) //weight initialization
            {
                for (int y = 0; y < 36; y++)
                {
                    weightMultiLayerPercep[x, y] = Math.Round(randomNumbers.NextDouble(), 2); //36 rows and 5 columns (weights including bias)
                }
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

            /*For character recognition*/
            outputCharacterRecognition = new List<int[]>{
                new int[] {0, 0, 0, 0, 0},
                new int[] {1, 0, 0, 0, 0},
                new int[] {0, 1, 0, 0, 0},
                new int[] {1, 1, 0, 0, 0},
                new int[] {0, 0, 1, 0, 0},
                new int[] {1, 0, 1, 0, 0},
                new int[] {0, 1, 1, 0, 0},
                new int[] {1, 1, 1, 0, 0},
                new int[] {0, 0, 0, 1, 0},
                new int[] {1, 0, 0, 1, 0},
                new int[] {0, 1, 0, 1, 0},
                new int[] {1, 1, 0, 1, 0},
                new int[] {0, 0, 1, 1, 0},
                new int[] {1, 0, 1, 1, 0},
                new int[] {0, 1, 1, 1, 0},
                new int[] {1, 1, 1, 1, 0},
                new int[] {0, 0, 0, 0, 1},
                new int[] {1, 0, 0, 0, 1},
                new int[] {0, 1, 0, 0, 1},
                new int[] {1, 1, 0, 0, 1},
                new int[] {0, 0, 1, 0, 1},
                new int[] {1, 0, 1, 0, 1},
                new int[] {0, 1, 1, 0, 1},
                new int[] {1, 1, 1, 0, 1},
                new int[] {0, 0, 0, 1, 1},
                new int[] {1, 0, 0, 1, 1}
            };

        }

        public bool addCharacterPattern(int [] pattern)
        {
            /*success*/
            trainingLetterPattern.Add(pattern);
            return true;
        }

        public bool addCharacterRecognitionPattern(int[] pattern)
        {
            /*success*/
            trainingCharacterRecogPattern.Add(pattern);
            return true;
        }

        public bool setTrainingLayerPatter(List<int[]> letters)
        {
            trainingLetterPattern.AddRange(letters);
            return true;
        }


        public bool setTrainingCharacterRecognition(List<int[]> letters)
        {
            trainingCharacterRecogPattern.AddRange(letters);
            return true;
        }

        public bool trainPerceptronCharacterRecognition()
        {
            try
            {
                int totalError = 1000;
                int arrayTotalError = 1000;
                double vValue = 0;
                int yValue = 0;
                int deltaValue = 0;
                int iteration = 0;
                int[] errorArray = new int[5];


                while (totalError != 0)
                {
                    iteration++;
                    totalError = 0;
                    arrayTotalError = 0;
                    for (int z = 0; z < 5; z++)
                    {
                        errorArray[z] = 0;
                    }


                    for (int i = 0; i < 26; i++)
                    {
                        for (int v = 0; v < 5; v++)
                        {
                            vValue = 0;
                            int errorPerV = 0;
                            //calculating v value
                            for (int j = 0; j < 36; j++)
                            {
                                if (j < 35)
                                    vValue += Math.Round(trainingLetterPattern[i][j] * weightMultiLayerPercep[v, j], 5);
                                else if (j == 35)
                                    vValue += weightMultiLayerPercep[v, j];
                            }
                            //calculating y value
                            if (vValue <= 0)
                            {
                                yValue = 0;
                            }
                            else if (vValue > 0)
                            {
                                yValue = 1;
                            }

                            //calculating delta value

                            deltaValue = outputCharacterRecognition[i][v] - yValue;

                            //calculating error
                            totalError += Math.Abs(deltaValue);
                            // errorArray[v] = errorPerV;

                            //recalculating weight
                            for (int x = 0; x < 36; x++)
                            {
                                if (x < 35)
                                    weightMultiLayerPercep[v, x] = Math.Round(weightMultiLayerPercep[v, x] + (learningRate * deltaValue * trainingLetterPattern[i][x]), 2);
                                else if (x == 35)
                                    weightMultiLayerPercep[v, x] = Math.Round(weightMultiLayerPercep[v, x] + (learningRate * deltaValue), 2);
                            }
                        }
                    }
                }
                return true;
            }
            catch(Exception ex)
            {
                return false; // training not successful
            }
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

                        /*evaluate the weighted sum using the threshold function*/
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

        public string whatLetter(int[] input)
        {
            int j = 0;
            double vValueResult = 0;
            StringBuilder sb = new StringBuilder();
            StringBuilder sbBinary = new StringBuilder();
            StringBuilder outputWeights = new StringBuilder();
            StringBuilder valueConversion = new StringBuilder();

            for (int x = 0; x < 5; x++)
            {
                vValueResult = 0;
                j = 0;
                foreach (var xVal in input)
                {
                    if (j < 35)
                    {
                        vValueResult += Math.Round(xVal * weightMultiLayerPercep[x,j], 5);
                        outputWeights.Append(weightMultiLayerPercep[x, j] + ", ");
                    }
                    j++;
                }
                vValueResult += weightMultiLayerPercep[x, 35];
                outputWeights.Append(weightMultiLayerPercep[x, j] + ", ");
                outputWeights.Append("\n");
                float valueToCompare = 0.0f;

                if (vValueResult > valueToCompare)
                {
                    sbBinary.Append("1");
                }
                else
                {
                    valueConversion.Append(vValueResult + " becomes 0\n");
                    sbBinary.Append("0");
                }
                sb.Append("v[" + x + "] = " + vValueResult + "\n");
            }
            return evaluateCharacterWeights(sbBinary.ToString());
        }

        public string evaluateCharacterWeights(string input)
        {
            string ret = "";
            if (input.Equals("00000"))
            {
                ret = "A";
            }
            else if (input.Equals("10000"))
            {
                ret = "B";
            }
            else if (input.Equals("01000"))
            {
                ret = "C";
            }
            else if (input.Equals("11000"))
            {
                ret = "D";
            }
            else if (input.Equals("00100"))
            {
                ret = "E";
            }
            else if (input.Equals("10100"))
            {
                ret = "F";
            }
            else if (input.Equals("01100"))
            {
                ret = "G";
            }
            else if (input.Equals("11100"))
            {
                ret = "H";
            }
            else if (input.Equals("00010"))
            {
                ret = "I";
            }
            else if (input.Equals("10010"))
            {
                ret = "J";
            }
            else if (input.Equals("01010"))
            {
                ret = "K";
            }
            else if (input.Equals("11010"))
            {
                ret = "L";
            }
            else if (input.Equals("00110"))
            {
                ret = "M";
            }
            else if (input.Equals("10110"))
            {
                ret = "N";
            }
            else if (input.Equals("01110"))
            {
                ret = "O";
            }
            else if (input.Equals("11110"))
            {
                ret = "P";
            }
            else if (input.Equals("00001"))
            {
                ret = "Q";
            }
            else if (input.Equals("10001"))
            {
                ret = "R";
            }
            else if (input.Equals("01001"))
            {
                ret = "S";
            }
            else if (input.Equals("11001"))
            {
                ret = "T";
            }
            else if (input.Equals("00101"))
            {
                ret = "U";
            }
            else if (input.Equals("10101"))
            {
                ret = "V";
            }
            else if (input.Equals("01101"))
            {
                ret = "W";
            }
            else if (input.Equals("11101"))
            {
                ret = "X";
            }
            else if (input.Equals("00011"))
            {
                ret = "Y";
            }
            else if (input.Equals("10011"))
            {
                ret = "Z";
            }

            return ret;
        }
    }
}
