using Microsoft.ML;
using HomeDepot.ML;

var mlContext = new MLContext(seed: 1);

var baseDataPath = Path.GetFullPath(Path.Combine("..", "..", "..", "..", "Data", "train.csv"));
var productDescriptionsPath = Path.GetFullPath(Path.Combine("..", "..", "..", "..", "Data", "product_descriptions.csv"));
var dataPath = Path.GetFullPath(Path.Combine("..", "..", "..", "..", "Data", "data.csv"));

// Transform data.
DataPreparer dataPreparer = new(mlContext);
dataPreparer.Run(baseDataPath, productDescriptionsPath, dataPath);
