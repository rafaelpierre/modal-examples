import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    from openai import OpenAI

    return (OpenAI,)


@app.cell
def _():
    url = input("Enter your Modal URL: ")
    return (url,)


@app.cell
def _(OpenAI, url):
    client = OpenAI(
        base_url=f"{url}/v1",
        api_key = "<YOUR_API_KEY>"
    )
    return (client,)


@app.cell
def _(client):
    client.chat.completions.create(
        max_tokens = 1000,
        top_p = 0.90,
        messages=[{"role": "user", "content": "What's ML?"}],
        model="google/gemma-4-26B-A4B-it")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
