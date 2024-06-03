using HomeDepot.Lib.ML;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;

namespace HomeDepot.ML
{
    internal class DataPreparer(MLContext mlContext)
    {
        private readonly MLContext _mlContext = mlContext;
        private readonly Dictionary<string, string> _productDescriptions = [];

        public void Run(string trainDataPath, string testDataPath, string productDescriptionsPath, string trainOutputPath, string testOutputPath)
        {
            if (!Path.Exists(trainDataPath))
                throw new ArgumentException("Unable to locate data.", nameof(trainDataPath));

            // Load complementary training data.
            LoadProductDescriptions(productDescriptionsPath);

            // Load base training data.
            IDataView trainDataView = _mlContext.Data.LoadFromTextFile<HomeDepotSample>(trainDataPath,
                hasHeader: true,
                separatorChar: ',',
                allowQuoting: true);

            // Load base test data.
            IDataView testDataView = _mlContext.Data.LoadFromTextFile<HomeDepotSample>(testDataPath,
                hasHeader: true,
                separatorChar: ',',
                allowQuoting: true);

            // Transform data.
            var pipeline = _mlContext.Transforms.CustomMapping(new ProductDescriptionMappingFactory(_productDescriptions).GetMapping(), contractName: Constants.ProductDescriptionColumnName);
            var transformedTrainData = pipeline.Fit(trainDataView).Transform(trainDataView);
            var transformedTestData = pipeline.Fit(testDataView).Transform(testDataView);

            // Save transformed data to file.
            using FileStream fileStream = new(trainOutputPath, FileMode.Create);
            _mlContext.Data.SaveAsText(transformedTrainData, fileStream, schema: false, separatorChar: ',');
            Console.WriteLine($"Wrote train data to: '{trainOutputPath}'");

            using FileStream fileStream2 = new(testOutputPath, FileMode.Create);
            _mlContext.Data.SaveAsText(transformedTestData, fileStream2, schema: false, separatorChar: ',');
            Console.WriteLine($"Wrote test data to: '{testOutputPath}'");
        }

        /// <summary>
        /// Populate <see cref="_productDescriptions"/> with data from file at <paramref name="path"/>.
        /// </summary>
        /// <param name="path">Path of CSV-file.</param>
        /// <exception cref="ArgumentException"></exception>
        /// <exception cref="InvalidOperationException"></exception>
        private void LoadProductDescriptions(string path)
        {
            if (!Path.Exists(path))
                throw new ArgumentException("Unable to locate product descriptions.", nameof(path));

            IDataView productDescriptionDataView = _mlContext.Data.LoadFromTextFile(path,
                hasHeader: true,
                separatorChar: ',',
                allowQuoting: true,
                columns: [
                    new TextLoader.Column(Constants.ProductUidColumnName,DataKind.String, 0),
                    new TextLoader.Column(Constants.ProductDescriptionColumnName,DataKind.String, 1)
                ]);
            foreach (var row in productDescriptionDataView.Preview(Int32.MaxValue).RowView)
            {
                string? uid = null, description = null;
                foreach (var column in row.Values)
                {
                    if (column.Key == Constants.ProductUidColumnName)
                    {
                        uid = column.Value.ToString();
                    }
                    else if (column.Key == Constants.ProductDescriptionColumnName)
                    {
                        description = column.Value.ToString();
                    }
                }

                // Verify values.
                if (string.IsNullOrEmpty(uid) || string.IsNullOrEmpty(description))
                    throw new InvalidOperationException($"Invalid product description row ({Constants.ProductUidColumnName}: {uid}, {Constants.ProductDescriptionColumnName}: {description?.Take(32)}{(description?.Length > 32 ? "..." : string.Empty)})");

                _productDescriptions.Add(uid, description);
            }
        }

        // https://learn.microsoft.com/en-us/dotnet/api/microsoft.ml.custommappingcatalog.custommapping?view=ml-dotnet
        [CustomMappingFactoryAttribute(Constants.ProductDescriptionColumnName)]
        private class ProductDescriptionMappingFactory(Dictionary<string, string> productDescriptions) : CustomMappingFactory<HomeDepotSample, ComplementaryData>
        {
            private readonly Dictionary<string, string> _productDescriptions = productDescriptions;

            public override Action<HomeDepotSample, ComplementaryData> GetMapping() =>
                GetComplementaryData;

            /// <summary>
            /// Get <see cref="ComplementaryData"/> for a given <see cref="HomeDepotSample"/>.
            /// </summary>
            /// <param name="input"><see cref="HomeDepotSample"/> which ID's will be used to get additional data.</param>
            /// <param name="output">Empty <see cref="ComplementaryData"/> object to populate with data.</param>
            /// <exception cref="InvalidOperationException"></exception>
            private void GetComplementaryData(HomeDepotSample input, ComplementaryData output)
            {
                if (!_productDescriptions.TryGetValue(input.ProductUid.ToString(), out var productDescription) ||
                    string.IsNullOrWhiteSpace(productDescription))
                    throw new InvalidOperationException($"Product description for product '{input.ProductUid}' was null or whitespace.");

                output.ProductDescription = productDescription;
            }
        }

        /// <summary>
        /// Base training data.
        /// </summary>
        private class HomeDepotSample
        {
            [LoadColumn(0)]
            [ColumnName(Constants.IdColumnName)]
            public int Id { get; set; }

            [LoadColumn(1)]
            [ColumnName(Constants.ProductUidColumnName)]
            public string ProductUid { get; set; } = string.Empty;

            [LoadColumn(2)]
            [ColumnName(Constants.ProductTitleColumnName)]
            public string ProductTitle { get; set; } = string.Empty;

            [LoadColumn(3)]
            [ColumnName(Constants.SearchTermColumnName)]
            public string SearchTerm { get; set; } = string.Empty;

            [LoadColumn(4)]
            [ColumnName(Constants.RelevanceColumnName)]
            public float Relevance { get; set; }
        }

        /// <summary>
        /// Additional data to complement <see cref="HomeDepotSample"/>.
        /// </summary>
        private class ComplementaryData
        {
            [ColumnName(Constants.ProductDescriptionColumnName)]
            public string ProductDescription { get; set; } = string.Empty;
        }
    }
}
