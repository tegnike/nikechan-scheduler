import argparse
import datetime
import logging
import os
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from pdf_processor import PDFProcessor
from text2pptx import Presentation, Slide, TextToPptx

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class PDF2PPTXState(BaseModel):
    # モデルの設定を追加
    model_config = {"arbitrary_types_allowed": True}  # カスタム型を許可

    # 出力
    output_path: Path = Field(..., description="出力PPTXのパス")

    # 入力関連
    pdf_path: Path = Field(..., description="入力PDFのパス")
    template_path: Path = Field(..., description="テンプレートPPTXのパス")

    # 中間データ
    pdf_pages: List[str] = Field(default_factory=list, description="PDFから抽出したテキスト")
    image_info: List[Dict[str, Any]] = Field(
        default_factory=list, description="PDFから抽出した画像情報"
    )

    # スライド生成関連
    slide_number: int = Field(default=1, description="生成するスライドの番号")
    pptx_structure: dict[str, Any] = Field(default_factory=dict, description="PPTXの構造")
    tmp_slide: Slide = Field(
        default_factory=lambda: Slide(type="empty", placeholders=[]),
        description="生成された仮スライド",
    )
    presentation: Presentation = Field(
        default_factory=lambda: Presentation(slides=[]),
        description="生成されたプレゼンテーション",
    )
    pass_slide_check: bool = Field(default=False, description="スライドの内容が正しいかどうか")
    all_slide_created: bool = Field(default=False, description="全てのスライドが生成されたかどうか")

    # 出力関連
    pptx_buffer: Optional[BytesIO] = Field(default=None, description="生成されたPPTXのバッファ")


def pdf_extraction_node(state: PDF2PPTXState) -> dict[str, Any]:
    """PDFからテキストを抽出するノード"""
    logger.info("PDFテキスト抽出を開始します...")
    loader = PyPDFLoader(str(state.pdf_path))
    pages = []
    for page in loader.lazy_load():
        pages.append(page.page_content)
    return {"pdf_pages": pages}


def image_extraction_node(state: PDF2PPTXState) -> dict[str, Any]:
    """PDFから画像情報を抽出するノード"""
    logger.info("PDF画像抽出を開始します...")
    pdf_processor = PDFProcessor(
        endpoint=os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT"),
        key=os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY"),
        output_dir=state.output_path / "images",
    )
    pdf_processor.process_pdf(source=str(state.pdf_path))
    return {"image_info": pdf_processor.get_results()["image_info"]}


def pptx_structure_extraction_node(state: PDF2PPTXState) -> dict[str, Any]:
    """PPTXの構造を抽出するノード"""
    logger.info("PPTX構造抽出を開始します...")
    return {"pptx_structure": TextToPptx().get_pptx_structure(state.template_path)}


def slide_generation_node(state: PDF2PPTXState) -> dict[str, Any]:
    """各スライドを生成するノード"""
    logger.info(f"{state.slide_number}枚目のスライドの生成を開始します...")
    text_to_pptx = TextToPptx()
    slide: Slide = text_to_pptx.generate_slide(
        pptx_structure=state.pptx_structure,
        slide_number=state.slide_number,
        text="\n".join(state.pdf_pages),
        image_info=state.image_info,
    )
    return {
        "tmp_slide": slide,
    }


def slide_check_node(state: PDF2PPTXState) -> dict[str, Any]:
    """スライドの内容が正しいかを判断するノード"""
    logger.info(f"{state.slide_number}枚目のスライドの内容をチェックします...")
    if TextToPptx().check_slide(state.tmp_slide):
        logger.info(f"{state.slide_number}枚目のスライドの内容は正しいです。")
        updated_presentation = Presentation(
            slides=state.presentation.slides + [state.tmp_slide]
        )
        return {
            "slide_number": state.slide_number + 1,
            "presentation": updated_presentation,
            "pass_slide_check": True,
        }
    else:
        logger.info(f"{state.slide_number}枚目のスライドの内容は正しくありません。")
        return {
            "pass_slide_check": False,
        }


def slide_count_node(state: PDF2PPTXState) -> dict[str, Any]:
    """すべてのスライド生成を完了したかどうかを判断するノード"""
    logger.info("すべてのスライド生成を完了したかどうかを判断します...")
    return {
        "all_slide_created": state.slide_number - 1
        >= len(state.pptx_structure["slides"])
    }


def pptx_conversion_node(state: PDF2PPTXState) -> dict[str, Any]:
    """最終的なPPTXを生成するノード"""
    logger.info("PPTXファイルの生成を開始します...")
    text_to_pptx = TextToPptx()
    pptx = text_to_pptx.convert(
        template_path=state.template_path,
        presentation=state.presentation,
    )
    return {"pptx_buffer": pptx}


class PaperToPptx:
    def __init__(self):
        self.workflow = StateGraph(PDF2PPTXState)
        self._build_graph()

    def _build_graph(self):
        # ノードの追加
        self.workflow.add_node("pdf_extraction", pdf_extraction_node)
        self.workflow.add_node("image_extraction", image_extraction_node)
        self.workflow.add_node(
            "pptx_structure_extraction", pptx_structure_extraction_node
        )
        self.workflow.add_node("slide_generation", slide_generation_node)
        self.workflow.add_node("slide_check", slide_check_node)
        self.workflow.add_node("slide_count", slide_count_node)
        self.workflow.add_node("pptx_conversion", pptx_conversion_node)

        # エントリーポイントの設定
        self.workflow.set_entry_point("pdf_extraction")

        # エッジの追加
        self.workflow.add_edge("pdf_extraction", "image_extraction")
        self.workflow.add_edge("image_extraction", "pptx_structure_extraction")
        self.workflow.add_edge("pptx_structure_extraction", "slide_generation")
        self.workflow.add_edge("slide_generation", "slide_check")

        # 条件付きエッジ
        self.workflow.add_conditional_edges(
            "slide_check",
            lambda state: state.pass_slide_check,
            {True: "slide_count", False: "slide_generation"},
        )

        self.workflow.add_conditional_edges(
            "slide_count",
            lambda state: state.all_slide_created,
            {True: "pptx_conversion", False: "slide_generation"},
        )

        # 最終エッジ
        self.workflow.add_edge("pptx_conversion", END)

    def run(self, source: str, template_path: str) -> str:
        """グラフを実行する"""
        # グラフのコンパイル
        app = self.workflow.compile()
        output_path = Path(
            f"output/{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}/"
        )
        output_path.mkdir(parents=True, exist_ok=False)

        # pdfのダウンロード
        if source.startswith("http://") or source.startswith("https://"):
            pdf_path = Path(output_path / "pdf.pdf")
            logger.info(f"PDFをダウンロードします: {source}")
            with open(pdf_path, "wb") as f:
                f.write(requests.get(source).content)
        else:
            pdf_path = Path(source)

        # 初期状態の設定
        initial_state = PDF2PPTXState(
            output_path=output_path,
            pdf_path=pdf_path,
            template_path=Path(template_path),
        )

        # グラフの実行
        final_state = app.invoke(initial_state)

        # pptx_bufferの存在チェック
        if "pptx_buffer" in final_state and final_state["pptx_buffer"]:
            # ファイルの保存
            output_path = final_state["output_path"] / "output.pptx"
            with open(output_path, "wb") as f:
                f.write(final_state["pptx_buffer"].getvalue())

            graph = app.get_graph()
            graph.draw_png(str(final_state["output_path"] / "workflow_graph.png"))

            return f"PowerPointファイルを保存しました: {output_path}"

        return "処理は完了しましたが、出力ファイルが生成されませんでした。"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PDFからPPTXを生成するツール")
    parser.add_argument("--source", required=True, help="PDFファイルのパスまたはURL")
    parser.add_argument(
        "--template", default="assets/template.pptx", help="テンプレートPPTXのパス"
    )
    args = parser.parse_args()

    graph = PaperToPptx()
    result = graph.run(
        source=args.source,
        template_path=args.template,
    )
    logger.info(result)
