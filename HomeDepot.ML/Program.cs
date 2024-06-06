using Microsoft.ML;
using HomeDepot.ML;
using Microsoft.ML.Data;
using Microsoft.ML.TorchSharp;
using Microsoft.ML.Transforms;
using HomeDepot.Lib.ML;
using MathNet.Numerics.Statistics;
using System.Diagnostics;

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
    new TextLoader.Column(Constants.ProductTitleColumnName, DataKind.String, 2),
    new TextLoader.Column(Constants.SearchTermColumnName, DataKind.String, 3),
    new TextLoader.Column(Constants.RelevanceColumnName, DataKind.Single, 4),
    new TextLoader.Column(Constants.ProductDescriptionColumnName, DataKind.String, 5),
    new TextLoader.Column(Constants.ProductInfoCombinedColumnName, DataKind.String, 6)
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
var sw = Stopwatch.StartNew();
var model = pipeline.Fit(dataSplit.TrainSet);
sw.Stop();

// Test model.
var predictions = model.Transform(dataSplit.TestSet);

// Evaluate trained model.
Evaluate(predictions, Constants.RelevanceColumnName, "Score", TimeSpan.FromMilliseconds(sw.ElapsedMilliseconds));

// Save model.
mlContext.Model.Save(model, data.Schema, "model.zip");

static void Evaluate(IDataView predictions, string actualColumnName, string predictedColumnName, TimeSpan duration)
{
    var actual = predictions.GetColumn<float>(actualColumnName).Select(x => (double)x);
    var predicted = predictions.GetColumn<float>(predictedColumnName).Select(x => (double)x);
    var results = predicted.Zip(actual, (p, a) => new { p, a });
    List<double> losses = [];
    ushort fails = 0;
    foreach (var result in results)
    {
        losses.Add(Math.Pow(result.p - result.a, 2));
        if (Math.Abs(result.p - result.a) > 0.5)
            fails++;
        Console.WriteLine($"Predicted: {result.p}\tActual: {result.a}");
    }
    var loss = losses.Average();
    var correlation = Correlation.Pearson(actual, predicted);
    var precision = 100 - (double)fails / results.Count() * 100;
    Console.WriteLine($"Loss: {loss} ({double.Round(loss, 4)})");
    Console.WriteLine($"Pearson Correlation: {correlation}");
    Console.WriteLine($"Precision: {precision}% ({double.Round(precision, 2)}%)");
    Console.WriteLine($"Duration: {duration:dd\\.hh\\:mm\\:ss}");
}
