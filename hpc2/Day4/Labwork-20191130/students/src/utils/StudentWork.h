#pragma once

class StudentWork {
public:
    StudentWork() = default;
    ~StudentWork() = default;
    StudentWork(const StudentWork&) = default;
    StudentWork& operator=(const StudentWork&) = default;
    
    virtual bool isImplemented() const = 0;
};