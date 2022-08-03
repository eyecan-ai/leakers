import numpy as np
from leakers.detectors.factory import RunesDetectorsFactory
from typing import List
import pydantic
import numpy as np
from airium import Airium
from pathlib import Path
import imageio
import rich
import click


class HTMLRuneElement(pydantic.BaseModel):
    filename: str
    code: List[float]
    idx: int

    def generate(self):
        a = Airium()
        with a.div():
            with a.div():
                a.img(src=self.filename)
            with a.div():
                a(f"Code: {self.code}")
            with a.div():
                a(f"Index: {self.idx}")
        return a

    @classmethod
    def build(cls, image: np.ndarray, code: List[float], idx: int, filename: str):
        dr = HTMLRuneElement(filename=filename, code=code, idx=idx)
        imageio.imwrite(filename, image)
        return dr


class HTMLRunePage:
    IMAGES_SUBFOLDER = "images"

    def __init__(self, folder: str) -> None:
        self._folder = Path(folder)
        self._images_folder = self._folder / self.IMAGES_SUBFOLDER
        if not self._images_folder.exists():
            self._images_folder.mkdir(exist_ok=True, parents=True)

    def generate(self, runes: List[dict]):
        a = Airium()
        with a.div():
            for rune in runes:
                image = rune["image"]
                code = rune["code"]
                idx = rune["id"]
                dr = HTMLRuneElement.build(
                    image, code, idx, str(self._images_folder / f"{idx}.png")
                )
                a(dr.generate())

        with open(self._folder / "index.html", "w") as f:
            f.write(str(a))

        rich.print("Runes page generated in:", self._folder)
        return str(self._folder / "index.html")


@click.command("export_html", help="Export runes html page")
@click.option("-m", "--model", required=True, help="Leakers Model File.")
@click.option("-o", "--output_folder", required=True, help="output folder")
@click.option("--cuda/--cpu", default=False, help="Cuda or CPU")
def export_html(
    model: str,
    output_folder: str,
    cuda: bool,
):

    import cv2
    from leakers.detectors.factory import RunesDetectorsFactory
    from leakers.boards.simple import RibbonImagesPoolBoard
    import rich
    import webbrowser

    device = "cuda" if cuda else "cpu"
    detector = RunesDetectorsFactory.create_from_checkpoint(
        filename=model, device=device
    )
    leakers = detector.generate_leakers(
        border=0,
        padding=0,
        output_size=256,
        batch_size=256,
    )

    page = HTMLRunePage(folder=output_folder)
    index_file = page.generate(leakers)

    webbrowser.open(index_file)


if __name__ == "__main__":
    export_html()
