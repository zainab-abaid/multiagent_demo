# How the SQL Tool Handles Database Schema

## Current Implementation

The SQL tool (`sql_tool_nl_to_sql()`) handles the database schema in the following way:

### Automatic Schema Inclusion

When you call `sql_tool_nl_to_sql()` with a natural language query:

1. **Schema Fetching**: The function automatically fetches the database schema using `db.get_table_info_no_throw()`
   - This gets all table names, column names, data types, and relationships
   - The schema is **cached** after the first fetch to avoid repeated database introspection

2. **Schema in Prompt**: The schema is included in the system prompt sent to the LLM:
   ```python
   system_prompt = f"""You are an assistant that answers questions by writing SQL queries...
   
   Database schema:
   {schema}
   
   When the user asks a question, you should:
   ...
   """
   ```

3. **SQL Generation**: The LLM uses the schema information to generate correct SQL queries with proper table and column names

### Schema Caching

- The schema is fetched once when the database is first initialized
- It's stored in `_schema_cache` to avoid repeated introspection calls
- This makes subsequent SQL generation faster

### Example Flow

```
User Query: "How many Latin tracks were sold in 2013?"
    ↓
sql_tool_nl_to_sql() called
    ↓
1. Initialize DB connection (if not already done)
2. Get cached schema (or fetch if first time)
3. Build prompt with schema + user query
4. LLM generates SQL using schema knowledge
5. Execute SQL query
6. Return results
```

## Why This Approach?

**Pros:**
- Simple: No need for separate schema-fetching steps in the plan
- Efficient: Schema is cached after first fetch
- Automatic: Schema is always available when generating SQL

**Cons:**
- Less explicit: The schema fetching happens "behind the scenes"
- Schema is included in every LLM call (adds tokens, but necessary for accuracy)

## Alternative: Separate Schema Step

If you wanted to make it more explicit, you could:

1. Add a `get_schema` tool that the planner can call first
2. Store the schema in the agent state
3. Pass it explicitly to SQL generation

However, the current approach is simpler and works well since:
- The schema is needed for every SQL query anyway
- Caching makes it efficient
- The LLM needs the schema context to generate correct SQL

## Schema Content

The schema includes:
- All table names (e.g., `Track`, `Album`, `Artist`, `Invoice`, `InvoiceLine`, etc.)
- Column names and data types for each table
- Primary keys and foreign key relationships
- Index information

This gives the LLM everything it needs to write correct SQL queries with proper joins and column references.

