using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.RazorPages;
using FraudDashboard.Models;
using System.Globalization;

namespace FraudDashboard.Pages.Transactions
{
	public class IndexModel : PageModel
	{
		[BindProperty(SupportsGet = true)]
		public int MinRisk { get; set; } = 0;

		[BindProperty(SupportsGet = true)]
		public string? Query { get; set; } = string.Empty; // optional, dacă vrei search server-side

		public List<Transaction> Transactions { get; set; } = new();

		public async Task OnGet()
		{
			ViewData["Title"] = "Transactions";
			ViewData["Subtitle"] = "Full list with fraud score";

			// exact același CSV pe care îl folosești în dashboard
			var csvPath = @"../1_database/fraud_transactions.csv";

			var lines = await System.IO.File.ReadAllLinesAsync(csvPath);

			if (lines.Length == 0)
			{
				Transactions = new List<Transaction>();

				// populate ViewData even if empty (so layout doesn't show --)
				ViewData["HighRiskToday"] = 0;
				ViewData["AvgRisk"] = "0%";
				return;
			}

			// map columns
			// We load the entire CSV into memory (keeps metrics correct), but we'll limit the
			// number of results shown to the user per filter to avoid rendering too many rows.
			var header = lines[0].Split(',');
			var headerMap = header
				.Select((name, idx) => new { name = name.Trim(), idx })
				.ToDictionary(x => x.name, x => x.idx, StringComparer.OrdinalIgnoreCase);

			int idx_time = headerMap["trans_date_trans_time"];
			int idx_merchant = headerMap["merchant"];
			int idx_category = headerMap["category"];
			int idx_amt = headerMap["amt"];
			int idx_city = headerMap["city"];
			int idx_state = headerMap["state"];
			int idx_trans_num = headerMap["trans_num"];
			int idx_is_fraud = headerMap["is_fraud"];

			var txAll = new List<Transaction>();

			// iterate over all data rows (keep full dataset in memory for metrics)
			for (int i = 1; i < lines.Length; i++)
			{
				var row = lines[i];
				if (string.IsNullOrWhiteSpace(row))
					continue;

				var cols = SplitCsvRow(row);

				string rawTime = idx_time < cols.Count ? cols[idx_time].Trim() : "";
				string rawMerchant = idx_merchant < cols.Count ? cols[idx_merchant].Trim() : "";
				string rawCategory = idx_category < cols.Count ? cols[idx_category].Trim() : "";
				string rawAmt = idx_amt < cols.Count ? cols[idx_amt].Trim() : "0";
				string rawCity = idx_city < cols.Count ? cols[idx_city].Trim() : "";
				string rawState = idx_state < cols.Count ? cols[idx_state].Trim() : "";
				string rawTransNum = idx_trans_num < cols.Count ? cols[idx_trans_num].Trim() : "";
				string rawIsFraud = idx_is_fraud < cols.Count ? cols[idx_is_fraud].Trim() : "0";

				rawMerchant = rawMerchant.Trim().Trim('"');

				decimal amountVal = 0m;
				decimal.TryParse(rawAmt, NumberStyles.Any, CultureInfo.InvariantCulture, out amountVal);

				DateTime timestampVal = DateTime.MinValue;
				if (!DateTime.TryParseExact(
						rawTime,
						"dd-MM-yyyy HH:mm",
						CultureInfo.InvariantCulture,
						DateTimeStyles.AssumeLocal,
						out timestampVal))
				{
					DateTime.TryParseExact(
						rawTime,
						"dd-MM-yyyy HH:mm:ss",
						CultureInfo.InvariantCulture,
						DateTimeStyles.AssumeLocal,
						out timestampVal
					);
				}

				double isFraudNum = 0.0;
				double.TryParse(rawIsFraud, NumberStyles.Any, CultureInfo.InvariantCulture, out isFraudNum);

				double riskPercent = isFraudNum > 0 ? 100.0 : 0.0;
				string riskLabel = isFraudNum > 0 ? "HIGH" : "OK";

				txAll.Add(new Transaction
				{
					Id = string.IsNullOrEmpty(rawTransNum) ? $"row_{i}" : rawTransNum,
					CardLast4 = "",
					Merchant = rawMerchant,
					Amount = amountVal,
					Currency = "USD",
					Timestamp = timestampVal,
					Risk = riskPercent,
					RiskLabel = riskLabel,
					Location = $"{rawCity}, {rawState}",
					Notes = rawCategory,
					Features = null
				});
			}

			// 1. aplicăm filtrul de risc minim (sliderul MinRisk din UI)
			var filtered = txAll
				.Where(t => t.Risk >= MinRisk)
				.ToList();

			// 2. aplicăm search server-side dacă Query nu e gol
			if (!string.IsNullOrWhiteSpace(Query))
			{
				var q = Query.Trim().ToLowerInvariant();
				filtered = filtered.Where(t =>
					(t.Merchant ?? string.Empty).ToLowerInvariant().Contains(q) ||
					(t.Id ?? string.Empty).ToLowerInvariant().Contains(q)
				).ToList();
			}

			// 3. sortare desc dupa timp (mai noi primele)
			// Show only up to 50 results after filtering to keep the page responsive
			Transactions = filtered
				.OrderByDescending(t => t.Timestamp)
				.Take(50)
				.ToList();

			//
			// 4. KPI pentru layout
			//
			// vrem să completăm ViewData["HighRiskToday"] și ViewData["AvgRisk"]
			// ca headerul global să nu mai arate "--" pe pagina Transactions

			// high risk today: aici facem o versiune simplă: câte tranzacții cu risc >=80% sunt în listă
			// (nu mai calculăm "ultima zi din dataset", ca pe dashboard; e ok să fie mai simplu aici)
			// calculam aceeasi metrica ca pe dashboard:

			// doar tranzactiile cu timestamp valid
			var validTsTx = txAll
				.Where(t => t.Timestamp != DateTime.MinValue)
				.ToList();

			int highRiskTodayForLayout = 0;

			if (validTsTx.Any())
			{
				// ultima zi existenta in dataset
				var lastDay = validTsTx.Max(t => t.Timestamp).Date;

				// cate tranzactii high-risk apar in acea zi
				highRiskTodayForLayout = validTsTx.Count(t =>
					t.Risk >= 80 &&
					t.Timestamp.Date == lastDay
				);
			}

			// media de risc pe toate tranzactiile
			var avgRiskVal = txAll.Any()
				? txAll.Average(t => t.Risk)
				: 0.0;

			// dam valori catre layout
			ViewData["HighRiskToday"] = highRiskTodayForLayout;
			ViewData["AvgRisk"] = $"{Math.Round(avgRiskVal, 1)}%";
		}

		// reuse the same CSV splitter you wrote before
		private static List<string> SplitCsvRow(string row)
		{
			var cols = new List<string>();
			bool inQuotes = false;
			var current = new System.Text.StringBuilder();

			for (int c = 0; c < row.Length; c++)
			{
				char ch = row[c];

				if (ch == '"')
				{
					if (inQuotes && c + 1 < row.Length && row[c + 1] == '"')
					{
						current.Append('"');
						c++;
					}
					else
					{
						inQuotes = !inQuotes;
					}
				}
				else if (ch == ',' && !inQuotes)
				{
					cols.Add(current.ToString());
					current.Clear();
				}
				else
				{
					current.Append(ch);
				}
			}

			cols.Add(current.ToString());
			return cols;
		}
	}
}
