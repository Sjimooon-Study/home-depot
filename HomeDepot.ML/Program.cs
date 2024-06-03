using Microsoft.ML;
using HomeDepot.ML;
using Microsoft.ML.Data;
using Microsoft.ML.TorchSharp;
using Microsoft.ML.Transforms;
using MathNet.Numerics.Statistics;
using HomeDepot.Lib.ML;

var mlContext = new MLContext(seed: 1);

var baseTrianDataPath = Path.GetFullPath(Path.Combine("..", "..", "..", "..", "Data", "train.csv"));
var baseTestDataPath = Path.GetFullPath(Path.Combine("..", "..", "..", "..", "Data", "test.csv"));
var productDescriptionsPath = Path.GetFullPath(Path.Combine("..", "..", "..", "..", "Data", "product_descriptions.csv"));
var trainDataPath = Path.GetFullPath(Path.Combine("..", "..", "..", "..", "Data", "train-data.csv"));
var testDataPath = Path.GetFullPath(Path.Combine("..", "..", "..", "..", "Data", "test-data.csv"));

// Transform data.
DataPreparer dataPreparer = new(mlContext);
dataPreparer.Run(baseTrianDataPath, baseTestDataPath, productDescriptionsPath, trainDataPath, testDataPath);

// Use GPU (Optional).
mlContext.GpuDeviceId = 0;
mlContext.FallbackToCpu = false;

// Log training output.
mlContext.Log += (o, e) =>
{
    if (e.Source.Contains("NasBertTrainer"))
        Console.WriteLine(e.Message);
};

// Load taining data.
var columns = new[]
{
    new TextLoader.Column(Constants.SearchTermColumnName, DataKind.String, 3),
    new TextLoader.Column(Constants.RelevanceColumnName, DataKind.Single, 4),
    new TextLoader.Column(Constants.ProductDescriptionColumnName, DataKind.String, 5)
};
var loaderOptions = new TextLoader.Options()
{
    Columns = columns,
    HasHeader = true,
    Separators = [','],
    MaxRows = 1000
};
var textLoader = mlContext.Data.CreateTextLoader(loaderOptions);
var data = textLoader.Load(trainDataPath);

// Split data (80% training, 20% testing).
var dataSplit = mlContext.Data.TrainTestSplit(data, testFraction: 0.2);

// Define training pipeline.
var pipeline = mlContext.Transforms
    .ReplaceMissingValues(Constants.RelevanceColumnName, replacementMode: MissingValueReplacingEstimator.ReplacementMode.Mean)
    .Append(mlContext.Regression.Trainers.SentenceSimilarity(
        labelColumnName: Constants.RelevanceColumnName,
        sentence1ColumnName: Constants.SearchTermColumnName,
        sentence2ColumnName: Constants.ProductDescriptionColumnName,
        batchSize: 32,
        maxEpochs: 10,
        architecture: Microsoft.ML.TorchSharp.NasBert.BertArchitecture.Roberta,
        validationSet: null));

// Train model.
var model = pipeline.Fit(dataSplit.TrainSet);

// Test model.
var predictions = model.Transform(dataSplit.TestSet);

// Evaluate trained model.
Evaluate(predictions, Constants.RelevanceColumnName, "Score");

// Save model.
mlContext.Model.Save(model, data.Schema, "model.zip");

static void Evaluate(IDataView predictions, string actualColumnName, string predictedColumnName)
{
    var actual = predictions.GetColumn<float>(actualColumnName).Select(x => (double)x);
    var predicted = predictions.GetColumn<float>(predictedColumnName).Select(x => (double)x);
    var results = predicted.Zip(actual, (p, a) => new { p, a });
    foreach (var result in results)
    {
        Console.WriteLine($"Predicted: {result.p}\tActual: {result.a}");
    }
    var correlation = Correlation.Pearson(actual, predicted);
    Console.WriteLine($"Pearson Correlation: {correlation}");
}
