# 📈 Monitoring Dashboard

## ✅ Current Monitoring

**Streamlit metrics display:**

- Model accuracy
- Training date
- Model name

## 📌 Manual Logs

- Printed in console from `train_high_performance.py`
- Includes accuracy, CV scores, best model

## 🛠 Future Plans

- Integrate Prometheus + Grafana
- Export logs to file or DB for dashboarding

---

## Current (Streamlit Metrics)

- Displays:
  - Predictions per day
  - Average response time
  - Success rate

> **Note**: These metrics are simulated. In production, integrate with:
> - **Grafana**: https://grafana.yourdomain.com
> - **Prometheus**: metrics endpoint at `/metrics`

## Future Enhancements

- Real-time logging of input data and predictions
- Error tracking (Sentry)
- User analytics for form submissions
