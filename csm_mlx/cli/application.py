from pathlib import Path

import typer
from huggingface_hub import hf_hub_download
from mlx_lm.sample_utils import make_sampler
from rich import print
from rich.progress import Progress, SpinnerColumn, TextColumn
from typing_extensions import Annotated

from csm_mlx import CSM, Segment, generate
from csm_mlx.cli.config import MODEL, Models
from csm_mlx.cli.utils import read_audio, write_audio

app = typer.Typer(no_args_is_help=True)


@app.command("generate")
def generate_command(
    text: str,
    output: Annotated[Path, typer.Option("--output", "-o")],
    model: Annotated[Models, typer.Option("--model", "-m")] = Models._1b,
    speaker: Annotated[int, typer.Option("--speaker", "-s")] = 0,
    max_audio_length: Annotated[int, typer.Option("--max-audio-length", "-l")] = 10000,
    temperature: Annotated[float, typer.Option("--temperature", "--temp", "-t")] = 0.8,
    top_p: Annotated[float | None, typer.Option("--top-p", "-p")] = None,
    min_p: Annotated[float | None, typer.Option("--min-p", "-m")] = 0.05,
    top_k: Annotated[int | None, typer.Option("--top-k", "-k")] = None,
    min_tokens_to_keep: Annotated[int, typer.Option("--min-tokens-to-keep", "-kt")] = 1,
    input_speakers: Annotated[
        list[int] | None, typer.Option("--input-speakers", "-is")
    ] = None,
    input_audios: Annotated[
        list[Path] | None, typer.Option("--input-audios", "-ia")
    ] = None,
    input_texts: Annotated[
        list[str] | None, typer.Option("--input-texts", "-it")
    ] = None,
):
    input_audios = input_audios or []
    input_texts = input_texts or []
    input_speakers = input_speakers or []

    if len(input_audios) != len(input_texts) or len(input_audios) != len(
        input_speakers
    ):
        print(
            "[bold red]Error![/bold red] All context inputs (input_audios, input_texts, and input_speakers) must have the same length."
        )
        raise typer.Exit(code=1)

    sampler = make_sampler(
        temp=temperature,
        top_p=top_p or 0.0,
        min_p=min_p or 0.0,
        top_k=top_k or -1,
        min_tokens_to_keep=min_tokens_to_keep,
    )

    # Download the weights!
    model_config = MODEL[model.value]
    sampling_rate = model_config.get("sampling_rate", 24000)

    weight = hf_hub_download(**model_config.get("loader"))  # type: ignore

    csm = CSM(model_config.get("config"))  # type: ignore
    csm.load_weights(weight)

    context = [
        Segment(speaker, text, read_audio(audio, sampling_rate))
        for audio, text, speaker in zip(input_audios, input_texts, input_speakers)
    ]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task(description="Inferencing...", total=None)
        result = generate(
            csm, text, speaker, context, max_audio_length, sampler=sampler
        )

    write_audio(result, output, sampling_rate)
    print(f"[bold green]Success![/bold green] Audio saved to: {output}")


if __name__ == "__main__":
    app()
