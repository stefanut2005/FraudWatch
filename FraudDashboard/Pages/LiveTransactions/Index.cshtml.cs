using Microsoft.AspNetCore.Mvc.RazorPages;

namespace FraudDashboard.Pages.LiveTransactions;

public class IndexModel : PageModel
{
    public void OnGet()
    {
        ViewData["Title"] = "Live Transactions";
        ViewData["Subtitle"] = "Real-time fraud detection";
    }
}

