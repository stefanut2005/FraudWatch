using Microsoft.AspNetCore.Mvc.RazorPages;
using FraudDashboard.Models;
using System.Globalization;

namespace FraudDashboard.Pages
{
	public class RiskBinPoint
	{
		public double ScoreBin { get; set; }
		public int Count { get; set; }
	}

	public class IndexModel : PageModel
	{
		public List<Transaction> Transactions { get; set; } = new();
		public List<RiskBinPoint> RiskHistogram { get; set; } = new();

		// KPI values (also shown in header)
		public int HighRiskToday { get; set; }
		public double AvgRisk { get; set; }

		public async Task OnGet()
		{
			ViewData["Title"] = "Overview";
			ViewData["Subtitle"] = "Live fraud risk across recent transactions";

			var csvPath = @"../1_database/fraud_transactions.csv";

			var lines = await System.IO.File.ReadAllLinesAsync(csvPath);

			if (lines.Length == 0)
			{
				Transactions = new List<Transaction>();
				RiskHistogram = new List<RiskBinPoint>();
				HighRiskToday = 0;
				AvgRisk = 0;

				// push defaults to layout too
				ViewData["HighRiskToday"] = HighRiskToday;
				ViewData["AvgRisk"] = $"{Math.Round(AvgRisk, 1)}%";
				return;
			}

			// map headers
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

			// parse CSV rows
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

				// merchant cleanup for quoted merchant names
				rawMerchant = rawMerchant.Trim().Trim('"');

				// amount
				decimal amountVal = 0m;
				decimal.TryParse(rawAmt, NumberStyles.Any, CultureInfo.InvariantCulture, out amountVal);

				// timestamp: try "dd-MM-yyyy HH:mm", then fallback "dd-MM-yyyy HH:mm:ss"
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

				// fraud flag -> risk%
				double isFraudNum = 0.0;
				double.TryParse(rawIsFraud, NumberStyles.Any, CultureInfo.InvariantCulture, out isFraudNum);

				double riskPercent = isFraudNum > 0 ? 100.0 : 0.0;
				string riskLabel = isFraudNum > 0 ? "HIGH" : "OK";

				Transactions.Add(new Transaction
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

			// histogram for chart
			RiskHistogram = Transactions
				.GroupBy(t => Math.Round(t.Risk / 100.0, 1))
				.Select(g => new RiskBinPoint
				{
					ScoreBin = g.Key,
					Count = g.Count()
				})
				.OrderBy(p => p.ScoreBin)
				.ToList();

			//
			// KPI calculations (safe)
			//

			// luăm doar tranzacțiile cu timestamp valid
			var txWithValidTs = Transactions
				.Where(t => t.Timestamp != DateTime.MinValue)
				.ToList();

			if (txWithValidTs.Any())
			{
				// ultima zi din dataset (nu azi din 2025)
				var maxTs = txWithValidTs.Max(t => t.Timestamp);

				HighRiskToday = txWithValidTs.Count(t =>
					t.Risk >= 80 &&
					t.Timestamp.Date == maxTs.Date
				);
			}
			else
			{
				HighRiskToday = 0;
			}

			AvgRisk = Transactions.Any()
				? Transactions.Average(t => t.Risk)
				: 0.0;

			//
			// expose KPI to layout header via ViewData
			//
			ViewData["HighRiskToday"] = HighRiskToday;
			ViewData["AvgRisk"] = $"{Math.Round(AvgRisk, 1)}%";
		}

		// CSV splitter that handles commas inside quotes and escaped quotes ("")
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
					// escaped double quote "" => literal "
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
