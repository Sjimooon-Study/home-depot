using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;

namespace HomeDepot.ML
{
    internal class DataPreparer(MLContext mlContext)
    {
        private readonly MLContext _mlContext = mlContext;
        private readonly Dictionary<string, string> _productDescriptions = [];

        public void Run(string dataPath, string productDescriptionsPath, string outputPath)
        {
            if (!Path.Exists(dataPath))
                throw new ArgumentException("Unable to locate data.", nameof(dataPath));

            // Load complementary training data.
            LoadProductDescriptions(productDescriptionsPath);

            // Load base training data.
            IDataView dataView = _mlContext.Data.LoadFromTextFile<HomeDepotSample>(dataPath,
                hasHeader: true,
                separatorChar: ',',
                allowQuoting: true);

            // Transform data.
            var pipeline = _mlContext.Transforms.CustomMapping(new ProductDescriptionMappingFactory(_productDescriptions).GetMapping(), contractName: Constants.ProductDescriptionColumnName);
            var transformedData = pipeline.Fit(dataView).Transform(dataView);

            // Save transformed data to file.
            using FileStream fileStream = new(outputPath, FileMode.Create);
            _mlContext.Data.SaveAsText(transformedData, fileStream, schema: false, separatorChar: ',');
            Console.WriteLine($"Wrote data to: '{outputPath}'");
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
            public string Relevance { get; set; } = string.Empty;
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
