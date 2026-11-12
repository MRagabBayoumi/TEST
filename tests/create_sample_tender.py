from pathlib import Path
from fpdf import FPDF
from PIL import Image, ImageDraw, ImageFont
import arabic_reshaper
from bidi.algorithm import get_display

OUTPUT_DIR = Path(__file__).parent / "samples"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_PDF = OUTPUT_DIR / "sample_tender.pdf"
SAMPLE_STAMP_IMG = OUTPUT_DIR / "sample_stamp.png"


def create_dummy_scanned_image():
    """Generates a realistic scanned document page image with Arabic text and stamps."""
    img = Image.new("RGB", (800, 1100), color=(250, 250, 250))
    draw = ImageDraw.Draw(img)

    try:
        font_large = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 26)
        font_regular = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 20)
    except Exception:
        font_large = font_regular = ImageFont.load_default()

    # Draw simulated official letterhead and stamp
    draw.rectangle([40, 40, 760, 1060], outline=(220, 220, 220), width=2)
    draw.ellipse([500, 750, 700, 950], outline=(180, 40, 40), width=4)

    # Arabic lines
    title_text = get_display(arabic_reshaper.reshape("محضر اجتماع لجنة التقييم الفني"))
    line1 = get_display(arabic_reshaper.reshape("بشأن مناقصة توريد الأغذية لمستشفيات جامعة القاهرة"))
    line2 = get_display(arabic_reshaper.reshape("تقرر استبعاد أي عطاء لا يقدم شهادة صحية سارية لجميع العاملين"))
    line3 = get_display(arabic_reshaper.reshape("الالتزام التام بالمطابقة للمواصفات القياسية المصرية رقم 2026"))

    draw.text((250, 100), title_text, fill=(20, 20, 20), font=font_large)
    draw.text((120, 180), line1, fill=(40, 40, 40), font=font_regular)
    draw.text((80, 260), line2, fill=(40, 40, 40), font=font_regular)
    draw.text((80, 320), line3, fill=(40, 40, 40), font=font_regular)

    img.save(SAMPLE_STAMP_IMG)
    return SAMPLE_STAMP_IMG


def generate_sample_pdf() -> Path:
    pdf = FPDF()
    font_path = "C:/Windows/Fonts/arial.ttf"

    pdf.add_font("ArialArabic", "", font_path)
    pdf.set_font("ArialArabic", size=13)

    # ---------------- PAGE 1: General Tender Information ----------------
    pdf.add_page()
    pdf.set_font("ArialArabic", size=16)
    pdf.cell(text="كراسة الشروط والمواصفات العامة", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(8)
    pdf.set_font("ArialArabic", size=12)
    pdf.multi_cell(
        w=0,
        h=8,
        text=(
            "المشروع: توريد الوجبات الغذائية المطهية لمستشفيات جامعة القاهرة\n"
            "رقم المناقصة: 14/2026\n"
            "مدة العقد: سنة ميلادية كاملة تبدأ من تاريخ توقيع العقد\n"
            "قيمة التأمين الابتدائي: 50,000 جنيه مصري\n"
            "آخر موعد لتقديم العطاءات: 25 نوفمبر 2026 الساعة 12 ظهراً"
        ),
        align="R",
    )

    # ---------------- PAGE 2: Food & Meal Specifications ----------------
    pdf.add_page()
    pdf.set_font("ArialArabic", size=16)
    pdf.cell(text="المواصفات الفنية للوجبات الغذائية", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(8)
    pdf.set_font("ArialArabic", size=12)
    pdf.multi_cell(
        w=0,
        h=8,
        text=(
            "أولاً: وجبة الإفطار\n"
            "- الصنف: جبن أبيض كامل الدسم\n"
            "- الكمية المقررة: 65 جرام لكل فرد\n"
            "- معدل التكرار: 7 أيام في الأسبوع (7/7)\n"
            "- التعبئة والتغليف: عبوات مغلفة آلياً ومحكمة الإغلاق\n\n"
            "ثانياً: وجبة الغداء\n"
            "- الصنف: دجاج مشوي\n"
            "- الكمية المقررة: 200 جرام وزن صافي بعد الطهي\n"
            "- معدل التكرار: 3 أيام في الأسبوع"
        ),
        align="R",
    )

    # ---------------- PAGE 3: Penalties and Disqualification ----------------
    pdf.add_page()
    pdf.set_font("ArialArabic", size=16)
    pdf.cell(text="الشروط القانونية والجزاءات", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(8)
    pdf.set_font("ArialArabic", size=12)
    pdf.multi_cell(
        w=0,
        h=8,
        text=(
            "1. في حالة تأخر المورد عن تسليم أي وجبة في موعدها المحدد، تطبق غرامة تأخير قدرها 1% عن كل ساعة تأخير.\n"
            "2. يستبعد العطاء تلقائياً إذا لم يرفق به أصل خطاب الضمان الابتدائي أو شيك مصرفي مقبول الدفع.\n"
            "3. يحق للجهة الإدارية فسخ العقد ومصادرة التأمين النهائي في حالة تكرار المخالفات ثلاث مرات متتالية."
        ),
        align="R",
    )

    # ---------------- PAGE 4: Simulated Scanned Page (Image only, no text layer) ----------------
    pdf.add_page()
    stamp_path = create_dummy_scanned_image()
    pdf.image(str(stamp_path), x=15, y=20, w=180)

    pdf.output(str(SAMPLE_PDF))
    return SAMPLE_PDF


if __name__ == "__main__":
    generate_sample_pdf()
    print(f"Generated sample PDF at: {SAMPLE_PDF}")
