create table public.customers (
  id uuid not null default gen_random_uuid (),
  created_at timestamp with time zone not null default now(),
  first_name text not null,
  last_name text null,
  email text not null,
  updated_at timestamp with time zone not null default now(),
  constraint customers_pkey primary key (id),
  constraint customers_email_key unique (email)
) TABLESPACE pg_default;

ALTER TABLE customers ENABLE ROW LEVEL SECURITY;

create table public.expenses (
  id uuid not null default gen_random_uuid (),
  created_at timestamp with time zone not null default now(),
  updated_at timestamp with time zone not null default now(),
  name text not null,
  description text null,
  category text not null default 'other'::text,
  amount real not null,
  customer_id uuid not null,
  constraint expenses_pkey primary key (id),
  constraint expenses_customer_id_fkey foreign KEY (customer_id) references customers (id) on update CASCADE on delete CASCADE
) TABLESPACE pg_default;

ALTER TABLE expenses ENABLE ROW LEVEL SECURITY;