from src.utils.display import print_trading_output


def test_print_trading_output_handles_empty_analyst_signals(capsys):
    result = {
        "decisions": {
            "AAPL": {
                "action": "hold",
                "quantity": 0,
                "confidence": 50.0,
                "reasoning": "No analyst data was available.",
            }
        },
        "analyst_signals": {},
    }

    print_trading_output(result)

    output = capsys.readouterr().out
    assert "No analyst signals available for AAPL" in output
    assert "TRADING DECISION" in output
    assert "PORTFOLIO SUMMARY" in output


def test_print_trading_output_renders_available_analyst_signal(capsys):
    result = {
        "decisions": {
            "AAPL": {
                "action": "buy",
                "quantity": 10,
                "confidence": 80.0,
                "reasoning": "Positive setup.",
            }
        },
        "analyst_signals": {
            "technical_analyst_agent": {
                "AAPL": {
                    "signal": "bullish",
                    "confidence": 75,
                    "reasoning": "Momentum is positive.",
                }
            }
        },
    }

    print_trading_output(result)

    output = capsys.readouterr().out
    assert "Technical Analyst" in output
    assert "BULLISH" in output
    assert "No analyst signals available" not in output
