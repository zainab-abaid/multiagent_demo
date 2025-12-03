# Sample Composite Query

## Query

```
What was the total revenue in USD from Latin tracks sold in 2013, and what would that amount be if converted to EUR based on the store's currency policy?
```

## Expected Agent Behavior

This query requires the agent to use all three tools:

### 1. SQL Tool (Database Query)
- Query the Chinook database to find:
  - All tracks in the "Latin" genre
  - InvoiceLine items for those tracks from invoices in 2013
  - Calculate total revenue (sum of UnitPrice * Quantity)
- Expected SQL query would join: Track -> Genre, InvoiceLine -> Invoice, filter by genre name "Latin" and invoice year 2013

### 2. RAG Tool (Document Retrieval)
- Search the documents folder for:
  - Currency conversion information
  - EUR to USD exchange rate from the pricing policy document
- Should retrieve information about the store's currency conversion rates (e.g., 1 EUR = 0.85 USD or 1 USD = 1.18 EUR)

### 3. API Tool (Currency Conversion)
- Use the `convert_to_usd` function (or reverse conversion) to:
  - Convert the USD revenue amount to EUR
  - Based on the exchange rate information from RAG

## Expected Output

When you run:
```bash
python debug_agent.py "What was the total revenue in USD from Latin tracks sold in 2013, and what would that amount be if converted to EUR based on the store's currency policy?"
```

The agent should:
1. **Plan** to execute all three tools in sequence:
   - Step 1: SQL tool to query revenue
   - Step 2: RAG tool to find currency policy
   - Step 3: API tool to convert currency
   - Step 4: Generate answer

2. **Execute SQL** to get the revenue amount
   - Shows: `[tool_caller:tool_call] tool: sql_tool latency: Xms`
   - **Correct SQL Query**: Should sum `UnitPrice * Quantity` from InvoiceLine for Latin tracks in 2013
   - **Expected Result**: $79.20 USD

3. **Execute RAG** to find currency conversion rates
   - Shows: `[tool_caller:tool_call] tool: rag_tool latency: Xms`
   - Retrieves EUR/USD rate from pricing_policy.txt (1 USD = 1.18 EUR)

4. **Execute API** to perform the conversion
   - Shows: `[tool_caller:tool_call] tool: convert_from_usd latency: Xms`
   - Converts USD amount to EUR using the rate from RAG

5. **Answer** should be:
   ```
   The total revenue from Latin tracks sold in 2013 was $79.20 USD. 
   Based on the store's currency policy (1 USD = 1.18 EUR), 
   this amount converts to approximately €93.46 EUR.
   ```

6. **Reflection** analyzes the episode execution

The debug output will show:
- Each step's events with latency and token counts
- Plan structure and current step cursor
- Latest tool results
- Final answer
- Trace summary with total tokens and latency

## Correct Answer (Verified from Database)

**Total Revenue (USD)**: $79.20  
**Total Revenue (EUR)**: €93.46 (using conversion rate 1 USD = 1.18 EUR)

**Note**: The SQL query should use:
```sql
SELECT SUM(il.UnitPrice * il.Quantity) AS TotalRevenueUSD
FROM InvoiceLine il
JOIN Invoice i ON il.InvoiceId = i.InvoiceId
JOIN Track t ON il.TrackId = t.TrackId
JOIN Genre g ON t.GenreId = g.GenreId
WHERE g.Name = 'Latin' AND strftime('%Y', i.InvoiceDate) = '2013';
```

**Important**: The query must sum `UnitPrice * Quantity` from InvoiceLine, NOT `Invoice.Total`, as the latter includes all items on the invoice, not just Latin tracks.

## Notes

- The exact revenue amount depends on the database content
- The conversion rate should come from the RAG-retrieved pricing policy document
- The API tool uses the rate: 1 EUR = 0.85 USD (so 1 USD = 1.18 EUR)

