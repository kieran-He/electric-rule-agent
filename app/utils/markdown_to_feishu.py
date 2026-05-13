import re


class MarkdownToFeishuConverter:
    def convert_to_interactive(self, text: str) -> dict:
        if not text or not text.strip():
            return self._empty_interactive()
        
        processed_text = self._process_latex_formulas(text)
        
        return {
            "schema": "2.0",
            "config": {
                "update_multi": True
            },
            "body": {
                "elements": [
                    {
                        "tag": "markdown",
                        "content": self._escape_for_card(processed_text)
                    }
                ]
            }
        }
    
    def _empty_interactive(self) -> dict:
        return {
            "schema": "2.0",
            "config": {
                "update_multi": True
            },
            "body": {
                "elements": [
                    {
                        "tag": "markdown",
                        "content": ""
                    }
                ]
            }
        }
    
    def _process_latex_formulas(self, text: str) -> str:
        block_pattern = re.compile(r'\$\$(.*?)\$\$', re.DOTALL)
        inline_pattern = re.compile(r'\$(.*?)\$')
        
        for match in list(block_pattern.finditer(text)):
            formula = match.group(1).strip()
            simple_formula = self._simplify_latex(formula)
            text = text.replace(match.group(0), simple_formula)
        
        for match in list(inline_pattern.finditer(text)):
            formula = match.group(1).strip()
            simple_formula = self._simplify_latex(formula)
            text = text.replace(match.group(0), simple_formula)
        
        return text
    
    def _simplify_latex(self, formula: str) -> str:
        formula = formula.replace('\\times', '×')
        formula = formula.replace('\\div', '÷')
        formula = formula.replace('\\pm', '±')
        formula = formula.replace('\\cdot', '·')
        formula = formula.replace('\\leq', '≤')
        formula = formula.replace('\\geq', '≥')
        formula = formula.replace('\\neq', '≠')
        formula = formula.replace('\\approx', '≈')
        formula = formula.replace('\\sum', '∑')
        formula = formula.replace('\\prod', '∏')
        formula = formula.replace('\\int', '∫')
        formula = formula.replace('\\sqrt', '√')
        formula = formula.replace('\\alpha', 'α')
        formula = formula.replace('\\beta', 'β')
        formula = formula.replace('\\gamma', 'γ')
        formula = formula.replace('\\delta', 'Δ')
        formula = formula.replace('\\epsilon', 'ε')
        formula = formula.replace('\\theta', 'θ')
        formula = formula.replace('\\lambda', 'λ')
        formula = formula.replace('\\mu', 'μ')
        formula = formula.replace('\\sigma', 'σ')
        formula = formula.replace('\\pi', 'π')
        formula = formula.replace('\\rho', 'ρ')
        formula = formula.replace('\\phi', 'φ')
        formula = formula.replace('\\omega', 'ω')
        
        fraction_pattern = re.compile(r'\\frac\{([^}]+)\}\{([^}]+)\}')
        for match in list(fraction_pattern.finditer(formula)):
            formula = formula.replace(match.group(0), f"{match.group(1)}/{match.group(2)}")
        
        formula = re.sub(r'[{}]', '', formula)
        formula = formula.replace('_', '')
        formula = formula.replace('^', '')
        
        return formula
    
    def _escape_for_card(self, text: str) -> str:
        text = text.replace('\\', '&#92;')
        return text