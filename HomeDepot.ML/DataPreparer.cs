using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;

namespace HomeDepot.ML
{
    internal class DataPreparer(MLContext mlContext)
    {
        private const string _idColumnName = "id";
        private const string _productUidColumnName = "product_uid";
        private const string _productTitleColumnName = "product_title";
        private const string _searchTermColumnName = "search_term";
        private const string _relevanceColumnName = "relevance";
        private const string _productDescriptionColumnName = "product_description";

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
            var pipeline = _mlContext.Transforms.CustomMapping(new ProductDescriptionMappingFactory(_productDescriptions).GetMapping(), contractName: _productDescriptionColumnName);
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
                    new TextLoader.Column(_productUidColumnName,DataKind.String, 0),
                    new TextLoader.Column(_productDescriptionColumnName,DataKind.String, 1)
                ]);
            foreach (var row in productDescriptionDataView.Preview(Int32.MaxValue).RowView)
            {
                string? uid = null, description = null;
                foreach (var column in row.Values)
                {
                    if (column.Key == _productUidColumnName)
                    {
                        uid = column.Value.ToString();
                    }
                    else if (column.Key == _productDescriptionColumnName)
                    {
                        description = column.Value.ToString();
                    }
                }

                // Verify values.
                if (string.IsNullOrEmpty(uid) || string.IsNullOrEmpty(description))
                    throw new InvalidOperationException($"Invalid product description row ({_productUidColumnName}: {uid}, {_productDescriptionColumnName}: {description?.Take(32)}{(description?.Length > 32 ? "..." : string.Empty)})");

                _productDescriptions.Add(uid, description);
            }
        }

        // https://learn.microsoft.com/en-us/dotnet/api/microsoft.ml.custommappingcatalog.custommapping?view=ml-dotnet
        [CustomMappingFactoryAttribute(_productDescriptionColumnName)]
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
            [ColumnName(_idColumnName)]
            public int Id { get; set; }

            [LoadColumn(1)]
            [ColumnName(_productUidColumnName)]
            public string ProductUid { get; set; } = string.Empty;

            [LoadColumn(2)]
            [ColumnName(_productTitleColumnName)]
            public string ProductTitle { get; set; } = string.Empty;

            [LoadColumn(3)]
            [ColumnName(_searchTermColumnName)]
            public string SearchTerm { get; set; } = string.Empty;

            [LoadColumn(4)]
            [ColumnName(_relevanceColumnName)]
            public string Relevance { get; set; } = string.Empty;
        }

        /// <summary>
        /// Additional data to complement <see cref="HomeDepotSample"/>.
        /// </summary>
        private class ComplementaryData
        {
            [ColumnName(_productDescriptionColumnName)]
            public string ProductDescription { get; set; } = string.Empty;
        }
    }
}
