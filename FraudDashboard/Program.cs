using System.Net.Http.Json;
using Microsoft.AspNetCore.Builder;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddRazorPages()
	.AddRazorRuntimeCompilation();

builder.Services.AddHttpClient("PythonApi", client =>
{
	var pythonApiBase = builder.Configuration["PythonApi:BaseUrl"]
						?? Environment.GetEnvironmentVariable("PYTHON_API_BASE")
						?? "http://localhost:5001";
	client.BaseAddress = new Uri(pythonApiBase);
});

builder.Services.AddCors(options =>
{
	options.AddPolicy("AllowAll", p =>
		p.AllowAnyOrigin()
		 .AllowAnyHeader()
		 .AllowAnyMethod());
});

var app = builder.Build();

if (!app.Environment.IsDevelopment())
{
	app.UseExceptionHandler("/Error");
	app.UseHsts();
	app.UseHttpsRedirection();
}
else
{
	app.Logger.LogInformation("Running in Development: HTTPS redirect is disabled.");
}

app.UseStaticFiles();
app.UseRouting();
app.UseCors("AllowAll");

app.MapGet("/proxy/transactions", async (IHttpClientFactory httpFactory) =>
{
	var client = httpFactory.CreateClient("PythonApi");
	// Read raw response first to surface truncated/invalid responses
	var resp = await client.GetAsync("/transactions");
	var raw = await resp.Content.ReadAsStringAsync();

	if (!resp.IsSuccessStatusCode)
	{
		app.Logger.LogError("Backend /transactions returned {StatusCode}: {Content}", resp.StatusCode, raw);
		return Results.Problem($"Failed to retrieve transactions: {resp.StatusCode}");
	}

	try
	{
		var result = System.Text.Json.JsonSerializer.Deserialize<List<object>>(raw);
		return Results.Json(result);
	}
	catch (Exception ex)
	{
		app.Logger.LogError(ex, "Failed to deserialize /transactions response. Raw content: {Raw}", raw);
		return Results.Problem("Invalid JSON returned from backend /transactions. Check backend logs for details.");
	}
});

app.MapPost("/proxy/chat", async (HttpRequest request, IHttpClientFactory httpFactory) =>
{
	var client = httpFactory.CreateClient("PythonApi");
	using var bodyReader = new StreamReader(request.Body);
	var rawBody = await bodyReader.ReadToEndAsync();

	using var forwardMsg = new HttpRequestMessage(HttpMethod.Post, "/chat")
	{
		Content = new StringContent(rawBody, System.Text.Encoding.UTF8, "application/json")
	};

	var resp = await client.SendAsync(forwardMsg);
	var payload = await resp.Content.ReadAsStringAsync();
	return Results.Content(payload, "application/json");
});

app.MapGet("/api/live/transaction", async (IHttpClientFactory httpFactory) =>
{
	try
	{
		var client = httpFactory.CreateClient();
		client.BaseAddress = new Uri("http://localhost:8000"); // MCP Server
		client.Timeout = TimeSpan.FromSeconds(30);
		
		var response = await client.GetAsync("/generate_and_predict_transaction");

		var raw = await response.Content.ReadAsStringAsync();

		if (!response.IsSuccessStatusCode)
		{
			app.Logger.LogError("MCP Server error: {StatusCode} - {Content}", response.StatusCode, raw);
			return Results.Problem($"Failed to generate transaction: {response.StatusCode} - see server logs");
		}

		try
		{
			var result = System.Text.Json.JsonSerializer.Deserialize<Dictionary<string, object>>(raw);
			return Results.Json(result);
		}
		catch (Exception ex)
		{
			app.Logger.LogError(ex, "Failed to deserialize /generate_and_predict_transaction response. Raw content: {Raw}", raw);
			// Return the raw string as fallback so the front-end can display it for debugging
			return Results.Content(raw, "application/json");
		}
	}
	catch (Exception ex)
	{
		app.Logger.LogError(ex, "Error in /api/live/transaction");
		return Results.Problem($"Error: {ex.Message}");
	}
});

app.MapRazorPages();
app.Run();
