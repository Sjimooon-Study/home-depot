using Microsoft.ML;
using HomeDepot.ML;
using Microsoft.ML.Data;
using Microsoft.ML.TorchSharp;
using Microsoft.ML.Transforms;
using MathNet.Numerics.Statistics;

var mlContext = new MLContext(seed: 1);

var baseDataPath = Path.GetFullPath(Path.Combine("..", "..", "..", "..", "Data", "train.csv"));
var productDescriptionsPath = Path.GetFullPath(Path.Combine("..", "..", "..", "..", "Data", "product_descriptions.csv"));
var dataPath = Path.GetFullPath(Path.Combine("..", "..", "..", "..", "Data", "data.csv"));

// Transform data.
DataPreparer dataPreparer = new(mlContext);
dataPreparer.Run(baseDataPath, productDescriptionsPath, dataPath);

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
    new TextLoader.Column("search_term", DataKind.String, 3),
    new TextLoader.Column("relevance", DataKind.Single, 4),
    new TextLoader.Column("product_description", DataKind.String, 5)
};
var loaderOptions = new TextLoader.Options()
{
    Columns = columns,
    HasHeader = true,
    Separators = [','],
    MaxRows = 1000
};
var textLoader = mlContext.Data.CreateTextLoader(loaderOptions);
var data = textLoader.Load(dataPath);

// Split data (80% training, 20% testing).
var dataSplit = mlContext.Data.TrainTestSplit(data, testFraction: 0.2);

// Define training pipeline.
var pipeline = mlContext.Transforms
    .ReplaceMissingValues("relevance", replacementMode: MissingValueReplacingEstimator.ReplacementMode.Mean)
    .Append(mlContext.Regression.Trainers.SentenceSimilarity(
        labelColumnName: "relevance",
        sentence1ColumnName: "search_term",
        sentence2ColumnName: "product_description",
        batchSize: 32,
        maxEpochs: 10,
        architecture: Microsoft.ML.TorchSharp.NasBert.BertArchitecture.Roberta,
        validationSet: null));

// Train model.
var model = pipeline.Fit(dataSplit.TrainSet);

// Test model.
var predictions = model.Transform(dataSplit.TestSet);

// Evaluate trained model.
Evaluate(predictions, "relevance", "Score");

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
