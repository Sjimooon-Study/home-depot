using Microsoft.ML;
using Microsoft.ML.Data;
using HomeDepot.Lib.ML;

string mlNetModelPath = Path.GetFullPath(Path.Combine("..", "..", "..", "..", "HomeDepot.ML", "bin", "Debug", "net8.0", "model.zip"));
//string mlNetModelPath = Path.GetFullPath("model.zip");
Lazy<PredictionEngine<ModelInput, ModelOutput>> predictEngine = new(CreatePredictEngine, true);

Console.WriteLine("Match description against a search term. Lowest score: 1. Highest score: 3.");
Console.WriteLine();

while (true)
{
    Console.WriteLine("Enter search term:");
    var searchTermInput = GetNonEmptyUserInput();

    Console.WriteLine("Enter product details:");
    var productDescriptionInput = GetNonEmptyUserInput();

    var modelInput = new ModelInput()
    {
        SearchTerm = searchTermInput,
        ProductDescription = productDescriptionInput
    };
    var modelOutput = Predict(modelInput);

    Console.WriteLine($"Score: {modelOutput.Score}");
    Console.WriteLine();
}

string GetNonEmptyUserInput()
{
    string? input;
    do
    {
        input = Console.ReadLine();
    } while (string.IsNullOrEmpty(input));

    return input;
}

PredictionEngine<ModelInput, ModelOutput> CreatePredictEngine()
{
    var mlContext = new MLContext();
    ITransformer mlModel = mlContext.Model.Load(mlNetModelPath, out DataViewSchema modelSchema);

    return mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(mlModel, modelSchema);
}

ModelOutput Predict(ModelInput input)
{
    var predEngine = predictEngine.Value;

    return predEngine.Predict(input);
}

class ModelInput
{
    [ColumnName(Constants.SearchTermColumnName)]
    public string SearchTerm { get; set; } = string.Empty;

    [ColumnName(Constants.ProductDescriptionColumnName)]
    public string ProductDescription { get; set; } = string.Empty;

    [ColumnName(Constants.RelevanceColumnName)]
    public float Relevance { get; set; }
}

class ModelOutput
{
    [ColumnName("Score")]
    public float Score { get; set; }
}
