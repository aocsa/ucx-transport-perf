#pragma once

enum class StatusCode : char {
  OK = 0,
  OutOfMemory = 1,
  KeyError = 2,
  TypeError = 3,
  Invalid = 4,
  IOError = 5,
  CapacityError = 6,
  IndexError = 7,
  UnknownError = 9,
  NotImplemented = 10,
  SerializationError = 11,
  RError = 13,
  CodeGenError = 40,
  ExpressionValidationError = 41,
  ExecutionError = 42,
  AlreadyExists = 45
};

struct Status {

  Status(StatusCode kind, std::string text)
      : kind(kind), text(text)
  {
  }

  StatusCode kind;
  std::string text;
};
