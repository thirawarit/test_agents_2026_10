---
title: Project Phoenix
owner: data-team
---

# โครงการ Phoenix

Project Phoenix ย้ายฐานข้อมูลลูกค้าจาก Oracle ไปยัง PostgreSQL 16 โดยทีม data engineering

## Backup และ Restore

ฐานข้อมูล PostgreSQL ใช้ pg_dump สำรองข้อมูลทุก 12 ชั่วโมง และเก็บไฟล์ไว้บน MinIO เป็นเวลา 14 วัน หลังการ migrate ทีมจะทดสอบ restore จาก snapshot ทุกเดือน

```sql
-- # ไม่ใช่หัวข้อ: คอมเมนต์ในบล็อกโค้ด
SELECT pg_size_pretty(pg_database_size('phoenix'));
```

## Timeline

Phase 1 เริ่มในเดือนมีนาคม ส่วน Phase 2 cutover มีกำหนดในเดือนกันยายน The rollback window stays open for 72 hours after cutover.

## ความเสี่ยง

ความเสี่ยงหลักคือ stored procedure ของ Oracle ที่ต้องเขียนใหม่ด้วย PL/pgSQL ทีมจึงทำ inventory ของ procedure ทั้งหมดก่อนเริ่ม Phase 2
