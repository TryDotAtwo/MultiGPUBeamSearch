import marimo

__generated_with = "0.13.15"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        """
        # Molab Runtime Probe

        This notebook probes the molab runtime, runs a small GPU matmul task,
        and then attempts a tiny CUDA build/run of the beam solver from GitHub.
        """
    )
    return


@app.cell
def _():
    # Paste/run the content of molab/probe_cell.py here if importing this file
    # does not preserve the script cell in molab.
    return


if __name__ == "__main__":
    app.run()
